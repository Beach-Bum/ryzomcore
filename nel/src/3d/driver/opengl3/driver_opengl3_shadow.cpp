// NeL - MMORPG Framework <https://wiki.ryzom.dev/>
// Copyright (C) 2026  Ryzom Core contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/**
 * Sun cascaded shadow maps (GL3 driver side).
 *
 * One GL_TEXTURE_2D_ARRAY depth texture holds all cascades (one layer each),
 * with one depth-only FBO per layer. The caster pass is a depth-only
 * re-render of the opaque scene driven by CRenderTrav::renderSunShadowCascades():
 * beginSunShadowCascade() binds the layer FBO, sets the viewport to the map
 * size, clears depth, disables colour writes and enables polygon offset; the
 * caller sets the light ortho camera through the regular setFrustum() /
 * setupViewMatrixEx() and renders; endSunShadowCascade() restores the state.
 *
 * The receive side lives in the mega pixel program (fogOrPpl=1 + cameraUBO=1
 * variants): the per-cascade matrices, splits and parameters travel in the
 * NlCamera UBO (see stageCameraUBO()), and the depth array is sampled through
 * a sampler2DArrayShadow bound to a reserved texture unit above the material
 * sampler range (see getSunShadowTextureUnit()).
 */

#include "stdopengl3.h"

#include "driver_opengl3.h"
#include "driver_opengl3_uniform_buffer.h"

#include <algorithm>
#include <cstring>

namespace NL3D {
namespace NLDRIVERGL3 {

// ***************************************************************************
sint CDriverGL3::getSunShadowTextureUnit() const
{
	// Material colour samplers always occupy units 0..IDRV_PROGRAM_MAXSAMPLERS-1
	// (see setupInitialUniforms()). The shadow sampler must live above that
	// range; when the hardware has no spare unit, sun shadows are unsupported.
	sint maxUnits = (sint)_Extensions.MaxFragmentTextureImageUnits;
	sint unit = std::min(maxUnits - 1, 15);
	if (unit < (sint)IDRV_PROGRAM_MAXSAMPLERS)
		return -1;
	return unit;
}

// ***************************************************************************
bool CDriverGL3::supportSunShadowMaps() const
{
	if (!m_UseMegaShaders)
		return false;
	if (!(m_UseMegaCameraUBO || m_UseMegaObjectUBO))
		return false;
	return getSunShadowTextureUnit() >= 0;
}

// ***************************************************************************
void CDriverGL3::bindSunShadowTexture(bool bind)
{
	sint unit = getSunShadowTextureUnit();
	if (unit < 0)
		return;
	// The reserved unit is outside the CDriverGLStates3 per-unit cache; talk to
	// GL directly and put the active unit back where the cache believes it is.
	nglActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D_ARRAY, (bind && _SunShadowTexId) ? _SunShadowTexId : 0);
	nglActiveTexture(GL_TEXTURE0 + _DriverGLStates.getActiveTexture());
}

// ***************************************************************************
void CDriverGL3::releaseSunShadowMaps()
{
	if (_SunShadowCasterPass)
		endSunShadowCascade();

	if (_SunShadowTexId)
	{
		bindSunShadowTexture(false);
		for (uint i = 0; i < NL_OPENGL3_MAX_SUN_SHADOW_CASCADES; ++i)
		{
			if (_SunShadowFBOs[i])
			{
				if (_DriverGLStates.getCurrBoundFramebuffer() == _SunShadowFBOs[i])
					_DriverGLStates.forceBindFramebuffer(0);
				nglDeleteFramebuffers(1, &_SunShadowFBOs[i]);
				_SunShadowFBOs[i] = 0;
			}
		}
		glDeleteTextures(1, &_SunShadowTexId);
		_SunShadowTexId = 0;
	}
	_SunShadowNumCascades = 0;
	_SunShadowMapSize = 0;
	_SunShadowReceive = false;
	_CameraUBODirty = true;
}

// ***************************************************************************
bool CDriverGL3::setupSunShadowMaps(uint numCascades, uint mapSize, uint pcfTaps)
{
	H_AUTO_OGL(CDriverGL3_setupSunShadowMaps)

	if (numCascades == 0)
	{
		releaseSunShadowMaps();
		return true;
	}

	if (!supportSunShadowMaps())
		return false;

	numCascades = std::min(numCascades, (uint)NL_OPENGL3_MAX_SUN_SHADOW_CASCADES);
	pcfTaps = (pcfTaps >= 5) ? 5 : (pcfTaps >= 3) ? 3 : 1;
	{
		GLint maxTexSize = 2048;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
		mapSize = NLMISC::raiseToNextPowerOf2(std::max(mapSize, 64u));
		mapSize = std::min(mapSize, (uint)std::max(maxTexSize, 64));
	}

	if (_SunShadowTexId && _SunShadowNumCascades == numCascades && _SunShadowMapSize == mapSize)
	{
		// Same allocation; only the PCF kernel may have changed
		if (_SunShadowPcfTaps != pcfTaps)
		{
			_SunShadowPcfTaps = pcfTaps;
			_CameraUBODirty = true;
		}
		return true;
	}

	releaseSunShadowMaps();

	sint unit = getSunShadowTextureUnit();
	nlassert(unit >= 0);

	// Depth array texture, one layer per cascade, hardware depth compare for sampler2DArrayShadow
	glGenTextures(1, &_SunShadowTexId);
	nglActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D_ARRAY, _SunShadowTexId);
	nglTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, (GLsizei)mapSize, (GLsizei)mapSize, (GLsizei)numCascades, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifdef USE_OPENGLES3
	// No border clamp in GLES 3.0; the shader treats samples beyond the cascade far plane as lit
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#else
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	{
		const GLfloat border[4] = { 1.f, 1.f, 1.f, 1.f }; // outside the map = lit
		glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, border);
	}
#endif
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	nglActiveTexture(GL_TEXTURE0 + _DriverGLStates.getActiveTexture());

	// One depth-only FBO per layer
	GLuint prevFBO = _DriverGLStates.getCurrBoundFramebuffer();
	bool ok = true;
	for (uint i = 0; i < numCascades; ++i)
	{
		nglGenFramebuffers(1, &_SunShadowFBOs[i]);
		_DriverGLStates.forceBindFramebuffer(_SunShadowFBOs[i]);
		nglFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _SunShadowTexId, 0, (GLint)i);
#ifdef USE_OPENGLES3
		{
			GLenum none = GL_NONE;
			glDrawBuffers(1, &none);
		}
#else
		glDrawBuffer(GL_NONE);
#endif
		glReadBuffer(GL_NONE);
		GLenum status = (GLenum)nglCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			nlwarning("GL3: Sun shadow cascade %u FBO incomplete (status 0x%x, map size %u)", i, (unsigned int)status, mapSize);
			ok = false;
		}
	}
	_DriverGLStates.forceBindFramebuffer(prevFBO);

	if (!ok)
	{
		releaseSunShadowMaps();
		return false;
	}

	_SunShadowNumCascades = numCascades;
	_SunShadowMapSize = mapSize;
	_SunShadowPcfTaps = pcfTaps;
	for (uint i = 0; i < NL_OPENGL3_MAX_SUN_SHADOW_CASCADES; ++i)
		_SunShadowSplits[i] = 0.f;
	std::memset(_SunShadowMatrix, 0, sizeof(_SunShadowMatrix));
	bindSunShadowTexture(true);
	_CameraUBODirty = true;

	nlinfo("GL3: Sun shadow maps allocated: %u cascades, %ux%u, PCF %ux%u, texture unit %d", numCascades, mapSize, mapSize, pcfTaps, pcfTaps, (int)unit);
	return true;
}

// ***************************************************************************
bool CDriverGL3::beginSunShadowCascade(uint cascade)
{
	H_AUTO_OGL(CDriverGL3_beginSunShadowCascade)

	if (!_SunShadowTexId || cascade >= _SunShadowNumCascades || _SunShadowCasterPass)
		return false;

	// Save the state we are about to change
	_SunShadowSavedFBO = _DriverGLStates.getCurrBoundFramebuffer();
	_DriverGLStates.getViewport(_SunShadowSavedViewport[0], _SunShadowSavedViewport[1], _SunShadowSavedViewport[2], _SunShadowSavedViewport[3]);
	_DriverGLStates.getColorMask(_SunShadowSavedColorMask[0], _SunShadowSavedColorMask[1], _SunShadowSavedColorMask[2], _SunShadowSavedColorMask[3]);
	_SunShadowSavedScissor = _DriverGLStates.isScissorTestEnabled();

	// The depth array becomes the render target: never leave it bound as a sampler source
	bindSunShadowTexture(false);
	_SunShadowCasterPass = true;
	_CameraUBODirty = true; // receive folds to off while the caster pass is active

	_DriverGLStates.bindFramebuffer(_SunShadowFBOs[cascade]);
	_DriverGLStates.viewport(0, 0, (sint)_SunShadowMapSize, (sint)_SunShadowMapSize);
	_DriverGLStates.enableScissorTest(false);
	_DriverGLStates.colorMask(false, false, false, false);
	_DriverGLStates.enableZWrite(true);
#ifdef USE_OPENGLES3
	glClearDepthf(1.f);
#else
	glClearDepth(1.0);
#endif
	glClear(GL_DEPTH_BUFFER_BIT);

	// Slope-scaled depth bias against self-shadowing acne
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.5f, 4.0f);

	return true;
}

// ***************************************************************************
void CDriverGL3::endSunShadowCascade()
{
	H_AUTO_OGL(CDriverGL3_endSunShadowCascade)

	if (!_SunShadowCasterPass)
		return;

	glDisable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(0.f, 0.f);

	_DriverGLStates.bindFramebuffer(_SunShadowSavedFBO);
	_DriverGLStates.viewport(_SunShadowSavedViewport[0], _SunShadowSavedViewport[1], _SunShadowSavedViewport[2], _SunShadowSavedViewport[3]);
	_DriverGLStates.colorMask(_SunShadowSavedColorMask[0], _SunShadowSavedColorMask[1], _SunShadowSavedColorMask[2], _SunShadowSavedColorMask[3]);
	_DriverGLStates.enableScissorTest(_SunShadowSavedScissor);

	_SunShadowCasterPass = false;
	bindSunShadowTexture(true);
	_CameraUBODirty = true;
}

// ***************************************************************************
void CDriverGL3::setSunShadowCascade(uint cascade, const NLMISC::CMatrix &lightViewProj, float splitFar)
{
	if (cascade >= NL_OPENGL3_MAX_SUN_SHADOW_CASCADES)
		return;
	std::memcpy(_SunShadowMatrix[cascade], lightViewProj.get(), 16 * sizeof(float));
	_SunShadowSplits[cascade] = splitFar;
	_CameraUBODirty = true;
}

// ***************************************************************************
void CDriverGL3::setSunShadowReceive(bool enable, float strength)
{
	_SunShadowReceive = enable && _SunShadowTexId != 0 && _SunShadowNumCascades != 0;
	NLMISC::clamp(strength, 0.f, 1.f);
	_SunShadowStrength = strength;
	if (_SunShadowReceive)
		bindSunShadowTexture(true);
	_CameraUBODirty = true;
}

} // NLDRIVERGL3
} // NL3D

/* end of file */
