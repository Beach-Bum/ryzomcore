/**
 * \file post_stack.h
 * \brief CPostStack
 * \date 2026-10-20
 * CPostStack — full-screen post-processing stack for the OpenGL3 driver
 */

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

#ifndef NL3D_POST_STACK_H
#define NL3D_POST_STACK_H
#include <nel/misc/types_nl.h>

// NeL includes
#include <nel/misc/geom_ext.h>

// Project includes
#include <nel/3d/u_material.h>

namespace NL3D {

class UDriver;
class UScene;
class ITexture;
class CTextureUser;
class CPixelProgram;
class CVertexProgram;
class CFXAA;

/**
 * \brief CPostStack
 * Post-processing pass chain applied in place on the default effect render target
 * (UDriver::beginDefaultRenderTarget / endDefaultRenderTarget), after the scene and before the UI.
 * GL3-only: shader sources are glsl300esf (UBO) / glsl330f; on drivers without those profiles
 * isSupported() is false and the caller keeps its legacy effects.
 *
 * Pass order (design docs/ryzom/research/post-stack-design.md): AO (part B, stub) -> bloom ->
 * colour grade -> FXAA -> resample sharpen. Every pass follows the CFXAA::applyEffect pattern:
 * fog off, 2D11 matrix, user VP/PP, one NlPost UBO (rcpFrame, projParams, aoParams, bloomParams,
 * gradeParams), scratch targets from CRenderTargetManager, recycled per pass.
 */
class CPostStack
{
public:
	CPostStack(NL3D::UDriver *driver);
	~CPostStack();

	/// True when the driver compiled the GL3 post programs (otherwise apply() is a no-op)
	bool isSupported() const { return m_Supported; }

	/** Tier table (WoW-named settings).
	 *	\param ao AmbientOcclusion 0..3 (part B; 0 = off)
	 *	\param aa AntiAliasing 0 none, 1 FXAA low (0.5/0.25/0.0833), 2 FXAA high (0.75/0.166/0.0833)
	 *	\param bloom Bloom on/off, bloomIntensity 0..1 (additive composite factor)
	 *	\param renderScale RenderScale 50..200 (%); the caller sizes the scene target with getScaledSize()
	 *	\param sharpness ResampleSharpness 0..1 (CAS-like sharpen, only when renderScale < 100)
	 *	\param grade ColorGradeStrength 0..1 (identity at 0)
	 */
	void setQuality(uint ao, uint aa, bool bloom, float bloomIntensity, uint renderScale, float sharpness, float grade);

	/// True if a user-facing stage is on (AO, AA, bloom or render scale != 100); colour grade alone does not run the stack
	bool anyStageEnabled() const;

	/// Scene target size for a window of windowW x windowH at the current render scale
	void getScaledSize(uint32 windowW, uint32 windowH, uint32 &w, uint32 &h) const;

	/** Run the enabled passes in place on the current render target, which must be the default effect
	 *	target (offscreen, managed by CRenderTargetManager, upload format Auto). The caller has already
	 *	reset the viewport and set the 2D11 matrix mode (same contract as CFXAA::applyEffect).
	 *	\param scene used for the camera frustum (depth linearisation params of the AO pass); may be NULL
	 */
	void apply(NL3D::UScene *scene);

	uint getAO() const { return m_AO; }
	uint getAA() const { return m_AA; }
	bool getBloom() const { return m_Bloom; }
	float getBloomIntensity() const { return m_BloomIntensity; }
	uint getRenderScale() const { return m_RenderScale; }
	float getSharpness() const { return m_Sharpness; }
	float getGradeStrength() const { return m_Grade; }

private:
	/// Uniform values of the NlPost UBO (std140: five vec4)
	struct CPostParams
	{
		float RcpFrame[4];    // 1/srcW, 1/srcH, srcW, srcH
		float ProjParams[4];  // near, far, projScaleX (2n/(r-l)), projScaleY (2n/(t-b))
		float AOParams[4];    // radius, strength, slices, steps (part B)
		float BloomParams[4]; // threshold, soft knee, intensity, 0
		float GradeParams[4]; // strength, S-curve amount, saturation, sharpness
		CPostParams();
	};

	void release();
	CPixelProgram *createPixelProgram(const char *name, const char *body);
	void setParams(CPixelProgram *pp, const CPostParams &p);
	/// Draw a full-screen quad with pp into the current render target, sampling tex0 (sampler0) and tex1 (sampler1)
	void drawPass(CPixelProgram *pp, ITexture *tex0, ITexture *tex1, UMaterial &mat, const CPostParams &p);
	/// In-place pass on the scene target (swapTextureHandle with a same-kind scratch, like CFXAA)
	void runInPlace(CPixelProgram *pp, ITexture *tex1, const CPostParams &p);
	/// Pass from tex0 (+tex1) into a different target; additive = blend one/one into dest
	void runTo(CPixelProgram *pp, ITexture *tex0, ITexture *tex1, CTextureUser *dest, bool additive, const CPostParams &p);
	void restoreSceneTarget();

	/// \name Passes
	// @{
	/// Part B stub: GTAO. Returns early. See bindSceneDepth() and CPostParams::ProjParams for the plumbing it uses.
	void applyAO();
	void applyBloom();
	void applyGrade();
	void applyFXAA();
	void applySharpen();
	// @}

	/** Depth plumbing for the AO pass (part B): bind/unbind the scene target's depth-stencil texture on the
	 *	driver's reserved `nlSceneDepth` sampler (IDriver::bindRenderTargetDepthTexture). Passes that sample it
	 *	must render into colour-only targets (CRenderTargetManager::getRenderTarget(w, h, false)).
	 *	\return false if the driver has no sampleable depth (then AO must be skipped)
	 */
	bool bindSceneDepth(bool bind);

private:
	UDriver *m_Driver;
	bool m_Supported;

	UMaterial m_Mat;    // replace (blend off)
	UMaterial m_MatAdd; // additive (one, one) for the bloom upsample chain
	NLMISC::CQuadUV m_QuadUV;
	CVertexProgram *m_VP;
	CPixelProgram *m_PPBright;
	CPixelProgram *m_PPDown;
	CPixelProgram *m_PPUp;
	CPixelProgram *m_PPComposite;
	CPixelProgram *m_PPGrade;
	CPixelProgram *m_PPSharpen;
	CFXAA *m_FXAA;

	// Quality
	uint m_AO;
	uint m_AA;
	bool m_Bloom;
	float m_BloomIntensity;
	uint m_RenderScale;
	float m_Sharpness;
	float m_Grade;

	// Per-apply state
	ITexture *m_SceneTarget;
	uint m_Width;
	uint m_Height;
	float m_ProjParams[4];

}; /* class CPostStack */

} /* namespace NL3D */

#endif /* #ifndef NL3D_POST_STACK_H */

/* end of file */
