/**
 * \file post_stack.cpp
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

#include "std3d.h"
#include "nel/3d/post_stack.h"

// STL includes
#include <algorithm>

// Project includes
#include "nel/3d/u_camera.h"
#include "nel/3d/u_scene.h"
#include "nel/3d/u_driver.h"
#include "nel/3d/vertex_program.h"
#include "nel/3d/pixel_program.h"
#include "nel/3d/material.h"
#include "nel/3d/texture_offscreen.h"
#include "nel/3d/texture_user.h"
#include "nel/3d/driver_user.h"
#include "nel/3d/u_texture.h"
#include "nel/3d/render_target_manager.h"
#include "nel/3d/uniform_buffer_format.h"
#include "nel/3d/uniform_buffer.h"
#include "nel/3d/frustum.h"
#include "nel/3d/fxaa.h"

using namespace std;

namespace NL3D {

namespace {

// ***************************************************************************
// Shader sources. GL3 only: glsl300esf (linked program, NlPost UBO) preferred,
// glsl330f (separate shader objects, plain uniforms) fallback. Both share the bodies.
// ***************************************************************************

const char *s_PostVPES =
	"#version 300 es\n"
	"precision highp float;\n"
	"precision highp int;\n"
	"layout(location = 0) in vec4 vposition;\n"
	"layout(location = 8) in vec4 vtexCoord0;\n"
	"smooth out vec4 texCoord0;\n"
	"// modelViewProjection from NlModel UBO\n"
	"void main()\n"
	"{\n"
	"  gl_Position = modelViewProjection * vposition;\n"
	"  texCoord0 = vec4(vtexCoord0.xy, 0.0, 0.0);\n"
	"}\n";

const char *s_PostVP330 =
	"#version 330\n"
	"#extension GL_ARB_separate_shader_objects : enable\n"
	"out gl_PerVertex { vec4 gl_Position; };\n"
	"layout(location = 0) in vec4 vposition;\n"
	"layout(location = 8) in vec4 vtexCoord0;\n"
	"layout(location = 8) smooth out vec4 texCoord0;\n"
	"uniform mat4 modelViewProjection;\n"
	"void main()\n"
	"{\n"
	"  gl_Position = modelViewProjection * vposition;\n"
	"  texCoord0 = vec4(vtexCoord0.xy, 0.0, 0.0);\n"
	"}\n";

const char *s_PostPPHeaderES =
	"#version 300 es\n"
	"precision highp float;\n"
	"precision highp int;\n"
	"out vec4 fragColor;\n"
	"smooth in vec4 texCoord0;\n"
	"uniform sampler2D sampler0;\n"
	"uniform sampler2D sampler1;\n"
	"uniform sampler2D nlSceneDepth;\n"
	"// rcpFrame, projParams, aoParams, bloomParams, gradeParams from the NlPost UBO\n";

const char *s_PostPPHeader330 =
	"#version 330\n"
	"#extension GL_ARB_separate_shader_objects : enable\n"
	"out vec4 fragColor;\n"
	"layout(location = 8) smooth in vec4 texCoord0;\n"
	"uniform sampler2D sampler0;\n"
	"uniform sampler2D sampler1;\n"
	"uniform sampler2D nlSceneDepth;\n"
	"uniform vec4 rcpFrame;\n"
	"uniform vec4 projParams;\n"
	"uniform vec4 aoParams;\n"
	"uniform vec4 bloomParams;\n"
	"uniform vec4 gradeParams;\n";

// Shared helpers
const char *s_PostCommon =
	"float nlLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }\n"
	// 13-tap downsample (Call of Duty: Advanced Warfare), px = source texel size
	"vec3 nlBox13(sampler2D s, vec2 uv, vec2 px)\n"
	"{\n"
	"  vec3 a = texture(s, uv + px * vec2(-2.0, -2.0)).rgb;\n"
	"  vec3 b = texture(s, uv + px * vec2( 0.0, -2.0)).rgb;\n"
	"  vec3 c = texture(s, uv + px * vec2( 2.0, -2.0)).rgb;\n"
	"  vec3 d = texture(s, uv + px * vec2(-2.0,  0.0)).rgb;\n"
	"  vec3 e = texture(s, uv).rgb;\n"
	"  vec3 f = texture(s, uv + px * vec2( 2.0,  0.0)).rgb;\n"
	"  vec3 g = texture(s, uv + px * vec2(-2.0,  2.0)).rgb;\n"
	"  vec3 h = texture(s, uv + px * vec2( 0.0,  2.0)).rgb;\n"
	"  vec3 i = texture(s, uv + px * vec2( 2.0,  2.0)).rgb;\n"
	"  vec3 j = texture(s, uv + px * vec2(-1.0, -1.0)).rgb;\n"
	"  vec3 k = texture(s, uv + px * vec2( 1.0, -1.0)).rgb;\n"
	"  vec3 l = texture(s, uv + px * vec2(-1.0,  1.0)).rgb;\n"
	"  vec3 m = texture(s, uv + px * vec2( 1.0,  1.0)).rgb;\n"
	"  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;\n"
	"}\n"
	// Linear eye depth from the non-linear [0,1] depth buffer value (projParams.x = near, .y = far). Part B (AO).
	"float nlLinearDepth(float d)\n"
	"{\n"
	"  float n = projParams.x; float f = projParams.y;\n"
	"  return (2.0 * n * f) / (f + n - (d * 2.0 - 1.0) * (f - n));\n"
	"}\n";

// Bloom bright pass: 13-tap downsample of the scene (full -> half res) with a soft-knee luma threshold
const char *s_PostBright =
	"void main()\n"
	"{\n"
	"  vec3 c = nlBox13(sampler0, texCoord0.xy, rcpFrame.xy);\n"
	"  float lum = nlLuma(c);\n"
	"  float th = bloomParams.x;\n"
	"  float knee = bloomParams.y;\n"
	"  float soft = clamp(lum - th + knee, 0.0, 2.0 * knee);\n"
	"  soft = soft * soft / (4.0 * knee + 1e-4);\n"
	"  float contrib = max(soft, lum - th) / max(lum, 1e-4);\n"
	"  fragColor = vec4(c * contrib, 1.0);\n"
	"}\n";

// Bloom downsample level
const char *s_PostDown =
	"void main()\n"
	"{\n"
	"  fragColor = vec4(nlBox13(sampler0, texCoord0.xy, rcpFrame.xy), 1.0);\n"
	"}\n";

// Bloom upsample: 3x3 tent (radius one source texel), blended additively into the larger level
const char *s_PostUp =
	"void main()\n"
	"{\n"
	"  vec2 uv = texCoord0.xy;\n"
	"  vec2 px = rcpFrame.xy;\n"
	"  vec3 c = texture(sampler0, uv + px * vec2(-1.0, -1.0)).rgb\n"
	"         + texture(sampler0, uv + px * vec2( 0.0, -1.0)).rgb * 2.0\n"
	"         + texture(sampler0, uv + px * vec2( 1.0, -1.0)).rgb\n"
	"         + texture(sampler0, uv + px * vec2(-1.0,  0.0)).rgb * 2.0\n"
	"         + texture(sampler0, uv).rgb * 4.0\n"
	"         + texture(sampler0, uv + px * vec2( 1.0,  0.0)).rgb * 2.0\n"
	"         + texture(sampler0, uv + px * vec2(-1.0,  1.0)).rgb\n"
	"         + texture(sampler0, uv + px * vec2( 0.0,  1.0)).rgb * 2.0\n"
	"         + texture(sampler0, uv + px * vec2( 1.0,  1.0)).rgb;\n"
	"  fragColor = vec4(c * (1.0 / 16.0), 1.0);\n"
	"}\n";

// Bloom composite: scene (sampler0) + bloom level 0 (sampler1) * intensity
const char *s_PostComposite =
	"void main()\n"
	"{\n"
	"  vec4 s = texture(sampler0, texCoord0.xy);\n"
	"  vec3 b = texture(sampler1, texCoord0.xy).rgb;\n"
	"  fragColor = vec4(clamp(s.rgb + b * bloomParams.z, 0.0, 1.0), s.a);\n"
	"}\n";

// Colour grade: mild S-curve (smoothstep blend) + saturation, no vignette; identity at strength 0
const char *s_PostGrade =
	"void main()\n"
	"{\n"
	"  vec4 s = texture(sampler0, texCoord0.xy);\n"
	"  vec3 c = s.rgb;\n"
	"  vec3 curve = c * c * (3.0 - 2.0 * c);\n"
	"  c = mix(c, curve, gradeParams.y);\n"
	"  float l = nlLuma(c);\n"
	"  c = mix(vec3(l), c, gradeParams.z);\n"
	"  c = mix(s.rgb, clamp(c, 0.0, 1.0), gradeParams.x);\n"
	"  fragColor = vec4(c, s.a);\n"
	"}\n";

// Resample sharpen (AMD FidelityFX CAS style), gradeParams.w = sharpness 0..1
const char *s_PostSharpen =
	"void main()\n"
	"{\n"
	"  vec2 uv = texCoord0.xy;\n"
	"  vec2 px = rcpFrame.xy;\n"
	"  vec3 a = texture(sampler0, uv + px * vec2(-1.0, -1.0)).rgb;\n"
	"  vec3 b = texture(sampler0, uv + px * vec2( 0.0, -1.0)).rgb;\n"
	"  vec3 c = texture(sampler0, uv + px * vec2( 1.0, -1.0)).rgb;\n"
	"  vec3 d = texture(sampler0, uv + px * vec2(-1.0,  0.0)).rgb;\n"
	"  vec4 e4 = texture(sampler0, uv);\n"
	"  vec3 e = e4.rgb;\n"
	"  vec3 f = texture(sampler0, uv + px * vec2( 1.0,  0.0)).rgb;\n"
	"  vec3 g = texture(sampler0, uv + px * vec2(-1.0,  1.0)).rgb;\n"
	"  vec3 h = texture(sampler0, uv + px * vec2( 0.0,  1.0)).rgb;\n"
	"  vec3 i = texture(sampler0, uv + px * vec2( 1.0,  1.0)).rgb;\n"
	"  vec3 mnRGB = min(min(min(d, e), min(f, b)), h);\n"
	"  vec3 mnRGB2 = min(mnRGB, min(min(a, c), min(g, i)));\n"
	"  mnRGB += mnRGB2;\n"
	"  vec3 mxRGB = max(max(max(d, e), max(f, b)), h);\n"
	"  vec3 mxRGB2 = max(mxRGB, max(max(a, c), max(g, i)));\n"
	"  mxRGB += mxRGB2;\n"
	"  vec3 rcpMRGB = 1.0 / max(mxRGB, vec3(1e-4));\n"
	"  vec3 ampRGB = clamp(min(mnRGB, 2.0 - mxRGB) * rcpMRGB, 0.0, 1.0);\n"
	"  ampRGB = sqrt(ampRGB);\n"
	"  float peak = -1.0 / mix(8.0, 5.0, clamp(gradeParams.w, 0.0, 1.0));\n"
	"  vec3 wRGB = ampRGB * peak;\n"
	"  vec3 rcpWeightRGB = 1.0 / (1.0 + 4.0 * wRGB);\n"
	"  vec3 o = clamp(((b + d + f + h) * wRGB + e) * rcpWeightRGB, 0.0, 1.0);\n"
	"  fragColor = vec4(o, e4.a);\n"
	"}\n";

// ***************************************************************************
// Ambient occlusion: GTAO main pass. Port of XeGTAO's main pass (GameTechDev/XeGTAO, MIT, see
// nel/src/3d/THIRD_PARTY.md) rewritten as a fragment shader; no depth MIP prefilter (level 0 of the
// scene depth is point-sampled), normals reconstructed in place (XE_GTAO_GENERATE_NORMALS_INPLACE),
// no bent normals, no thin-occluder heuristic, no temporal noise (static interleaved gradient).
// View space here: x right, y up, z = linear eye depth (forward). GL texture rows go up, so the
// slice direction in screen space equals the view-space xy direction (no y flip).
// Inputs: nlSceneDepth (full-res non-linear depth), rcpFrame = AO target texel size / size,
// projParams = (near, far, P00, P11), aoParams = (effect radius [m], AO texel / depth texel ratio, slices, steps).
// Output (RGBA8): R = visibility / XE_GTAO_OCCLUSION_TERM_SCALE (working term), G = packed LRTB edges.
// ***************************************************************************
const char *s_PostGTAO =
	"const float NL_GTAO_PI = 3.1415926535897932;\n"
	"const float NL_GTAO_PI_HALF = 1.5707963267948966;\n"
	"const float NL_GTAO_FALLOFF_RANGE = 0.615;\n"            // XE_GTAO_DEFAULT_FALLOFF_RANGE
	"const float NL_GTAO_SAMPLE_DISTRIBUTION_POWER = 2.0;\n"  // XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER
	"const float NL_GTAO_FINAL_VALUE_POWER = 2.2;\n"          // XE_GTAO_DEFAULT_FINAL_VALUE_POWER
	"const float NL_GTAO_OCCLUSION_TERM_SCALE = 1.5;\n"       // XE_GTAO_OCCLUSION_TERM_SCALE
	"const float NL_GTAO_PIXEL_TOO_CLOSE = 1.3;\n"
	"float nlGtaoFastACos(float inX)\n"
	"{\n"
	"  float x = abs(inX);\n"
	"  float res = -0.156583 * x + NL_GTAO_PI_HALF;\n"
	"  res *= sqrt(max(1.0 - x, 0.0));\n"
	"  return (inX >= 0.0) ? res : NL_GTAO_PI - res;\n"
	"}\n"
	// NDC -> view (symmetric frustum): x = ndc.x * z / P00, y = ndc.y * z / P11
	"vec3 nlGtaoViewPos(vec2 uv, float z)\n"
	"{\n"
	"  vec2 ndc = uv * 2.0 - 1.0;\n"
	"  return vec3(ndc.x * z / projParams.z, ndc.y * z / projParams.w, z);\n"
	"}\n"
	"float nlGtaoViewZ(vec2 uv) { return nlLinearDepth(texture(nlSceneDepth, uv).r); }\n"
	// XeGTAO_CalculateEdges
	"vec4 nlGtaoEdges(float centerZ, float leftZ, float rightZ, float topZ, float bottomZ)\n"
	"{\n"
	"  vec4 e = vec4(leftZ, rightZ, topZ, bottomZ) - centerZ;\n"
	"  float slopeLR = (e.y - e.x) * 0.5;\n"
	"  float slopeTB = (e.w - e.z) * 0.5;\n"
	"  vec4 eAdj = e + vec4(slopeLR, -slopeLR, slopeTB, -slopeTB);\n"
	"  e = min(abs(e), abs(eAdj));\n"
	"  return clamp(1.25 - e / (centerZ * 0.011), 0.0, 1.0);\n"
	"}\n"
	// XeGTAO_PackEdges (2 bits per edge in an 8-bit unorm)
	"float nlGtaoPackEdges(vec4 e)\n"
	"{\n"
	"  e = round(clamp(e, 0.0, 1.0) * 2.9);\n"
	"  return dot(e, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));\n"
	"}\n"
	// XeGTAO_CalculateNormal: 4-tap edge-weighted normal from the view-space neighbours
	"vec3 nlGtaoNormal(vec4 e, vec3 c, vec3 l, vec3 r, vec3 t, vec3 b)\n"
	"{\n"
	"  vec4 acc = clamp(vec4(e.x * e.z, e.z * e.y, e.y * e.w, e.w * e.x) + 0.01, 0.0, 1.0);\n"
	"  l = normalize(l - c); r = normalize(r - c); t = normalize(t - c); b = normalize(b - c);\n"
	"  vec3 n = acc.x * cross(l, t) + acc.y * cross(t, r) + acc.z * cross(r, b) + acc.w * cross(b, l);\n"
	"  return normalize(n);\n"
	"}\n"
	// Interleaved gradient noise (Jimenez 2014)
	"float nlIGN(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }\n"
	"void main()\n"
	"{\n"
	"  vec2 px = rcpFrame.xy;\n"
	// At half-res an AO texel centre sits exactly on a full-res depth texel border (NEAREST would pick either):
	// shift by half a depth texel so every lookup lands on a texel centre. aoParams.y = AO texel / depth texel (1 or 2).
	"  float depthScale = max(aoParams.y, 1.0);\n"
	"  vec2 uv = texCoord0.xy - 0.5 * px * (depthScale - 1.0) / depthScale;\n"
	"  float far = projParams.y;\n"
	"  float d = texture(nlSceneDepth, uv).r;\n"
	"  if (d >= 0.9999995)\n" // sky / far plane: fully visible, no edges
	"  {\n"
	"    fragColor = vec4(1.0 / NL_GTAO_OCCLUSION_TERM_SCALE, 0.0, 0.0, 1.0);\n"
	"    return;\n"
	"  }\n"
	"  float viewZ = nlLinearDepth(d);\n"
	"  float zL = nlGtaoViewZ(uv + vec2(-px.x, 0.0));\n"
	"  float zR = nlGtaoViewZ(uv + vec2( px.x, 0.0));\n"
	"  float zT = nlGtaoViewZ(uv + vec2(0.0,  px.y));\n"
	"  float zB = nlGtaoViewZ(uv + vec2(0.0, -px.y));\n"
	"  vec4 edges = nlGtaoEdges(viewZ, zL, zR, zT, zB);\n"
	"  float packedEdges = nlGtaoPackEdges(edges);\n"
	// distance fade: the sky dome / far landscape ends up fully visible
	"  float farFade = clamp((viewZ - 0.90 * far) / (0.07 * far), 0.0, 1.0);\n"
	"  if (farFade >= 1.0)\n"
	"  {\n"
	"    fragColor = vec4(1.0 / NL_GTAO_OCCLUSION_TERM_SCALE, packedEdges, 0.0, 1.0);\n"
	"    return;\n"
	"  }\n"
	"  vec3 C = nlGtaoViewPos(uv, viewZ);\n"
	"  vec3 L = nlGtaoViewPos(uv + vec2(-px.x, 0.0), zL);\n"
	"  vec3 R = nlGtaoViewPos(uv + vec2( px.x, 0.0), zR);\n"
	"  vec3 T = nlGtaoViewPos(uv + vec2(0.0,  px.y), zT);\n"
	"  vec3 B = nlGtaoViewPos(uv + vec2(0.0, -px.y), zB);\n"
	"  vec3 viewspaceNormal = nlGtaoNormal(edges, C, L, R, T, B);\n"
	// move the centre slightly towards the camera (24-bit depth imprecision)
	"  viewZ *= 0.9995;\n"
	"  vec3 pixCenterPos = nlGtaoViewPos(uv, viewZ);\n"
	"  vec3 viewVec = normalize(-pixCenterPos);\n"
	"  float effectRadius = aoParams.x;\n"
	// view-space size of one AO texel at this depth: z * (2 / P00) * px.x
	"  float pixelSizeView = viewZ * 2.0 / projParams.z * px.x;\n"
	"  float screenspaceRadius = effectRadius / pixelSizeView;\n"
	// screen-space radius clamp (25 % of the target height): shrink the world radius with it
	"  float maxRadius = rcpFrame.w * 0.25;\n"
	"  if (screenspaceRadius > maxRadius)\n"
	"  {\n"
	"    effectRadius *= maxRadius / screenspaceRadius;\n"
	"    screenspaceRadius = maxRadius;\n"
	"  }\n"
	"  float falloffRange = NL_GTAO_FALLOFF_RANGE * effectRadius;\n"
	"  float falloffFrom = effectRadius * (1.0 - NL_GTAO_FALLOFF_RANGE);\n"
	"  float falloffMul = -1.0 / falloffRange;\n"
	"  float falloffAdd = falloffFrom / falloffRange + 1.0;\n"
	// fade out for small screen radii
	"  float visibility = clamp((10.0 - screenspaceRadius) / 100.0, 0.0, 1.0) * 0.5;\n"
	"  float minS = NL_GTAO_PIXEL_TOO_CLOSE / screenspaceRadius;\n"
	"  vec2 pix = floor(gl_FragCoord.xy);\n"
	"  float noiseSlice = nlIGN(pix);\n"
	"  float noiseSample = nlIGN(pix + vec2(17.0, 31.0));\n"
	"  int sliceCount = int(aoParams.z);\n"
	"  int stepsPerSlice = int(aoParams.w);\n"
	"  for (int slice = 0; slice < sliceCount; ++slice)\n"
	"  {\n"
	"    float sliceK = (float(slice) + noiseSlice) / float(sliceCount);\n"
	"    float phi = sliceK * NL_GTAO_PI;\n"
	"    float cosPhi = cos(phi);\n"
	"    float sinPhi = sin(phi);\n"
	"    vec2 omega = vec2(cosPhi, sinPhi) * screenspaceRadius;\n"
	"    vec3 directionVec = vec3(cosPhi, sinPhi, 0.0);\n"
	"    vec3 orthoDirectionVec = directionVec - dot(directionVec, viewVec) * viewVec;\n"
	"    vec3 axisVec = normalize(cross(orthoDirectionVec, viewVec));\n"
	"    vec3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);\n"
	"    float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));\n"
	"    float projectedNormalVecLength = length(projectedNormalVec);\n"
	"    float cosNorm = clamp(dot(projectedNormalVec, viewVec) / max(projectedNormalVecLength, 1e-5), 0.0, 1.0);\n"
	"    float n = signNorm * nlGtaoFastACos(cosNorm);\n"
	"    float lowHorizonCos0 = cos(n + NL_GTAO_PI_HALF);\n"
	"    float lowHorizonCos1 = cos(n - NL_GTAO_PI_HALF);\n"
	"    float horizonCos0 = lowHorizonCos0;\n"
	"    float horizonCos1 = lowHorizonCos1;\n"
	"    for (int step = 0; step < stepsPerSlice; ++step)\n"
	"    {\n"
	"      float stepBaseNoise = float(slice + step * stepsPerSlice) * 0.6180339887498948482;\n"
	"      float stepNoise = fract(noiseSample + stepBaseNoise);\n"
	"      float s = (float(step) + stepNoise) / float(stepsPerSlice);\n"
	"      s = pow(s, NL_GTAO_SAMPLE_DISTRIBUTION_POWER);\n"
	"      s += minS;\n"
	"      vec2 sampleOffset = round(s * omega) * px;\n"
	"      vec2 sp0 = uv + sampleOffset;\n"
	"      vec2 sp1 = uv - sampleOffset;\n"
	"      vec3 samplePos0 = nlGtaoViewPos(sp0, nlGtaoViewZ(sp0));\n"
	"      vec3 samplePos1 = nlGtaoViewPos(sp1, nlGtaoViewZ(sp1));\n"
	"      vec3 sampleDelta0 = samplePos0 - pixCenterPos;\n"
	"      vec3 sampleDelta1 = samplePos1 - pixCenterPos;\n"
	"      float sampleDist0 = max(length(sampleDelta0), 1e-5);\n"
	"      float sampleDist1 = max(length(sampleDelta1), 1e-5);\n"
	"      vec3 sampleHorizonVec0 = sampleDelta0 / sampleDist0;\n"
	"      vec3 sampleHorizonVec1 = sampleDelta1 / sampleDist1;\n"
	"      float weight0 = clamp(sampleDist0 * falloffMul + falloffAdd, 0.0, 1.0);\n"
	"      float weight1 = clamp(sampleDist1 * falloffMul + falloffAdd, 0.0, 1.0);\n"
	"      float shc0 = dot(sampleHorizonVec0, viewVec);\n"
	"      float shc1 = dot(sampleHorizonVec1, viewVec);\n"
	"      shc0 = mix(lowHorizonCos0, shc0, weight0);\n"
	"      shc1 = mix(lowHorizonCos1, shc1, weight1);\n"
	"      horizonCos0 = max(horizonCos0, shc0);\n"
	"      horizonCos1 = max(horizonCos1, shc1);\n"
	"    }\n"
	"    projectedNormalVecLength = mix(projectedNormalVecLength, 1.0, 0.05);\n"
	"    float h0 = -nlGtaoFastACos(horizonCos1);\n"
	"    float h1 = nlGtaoFastACos(horizonCos0);\n"
	"    float iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;\n"
	"    float iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;\n"
	"    visibility += projectedNormalVecLength * (iarc0 + iarc1);\n"
	"  }\n"
	"  visibility /= float(sliceCount);\n"
	"  visibility = pow(max(visibility, 0.0), NL_GTAO_FINAL_VALUE_POWER);\n"
	"  visibility = max(0.03, visibility);\n"
	"  visibility = mix(visibility, 1.0, farFade);\n"
	"  fragColor = vec4(clamp(visibility / NL_GTAO_OCCLUSION_TERM_SCALE, 0.0, 1.0), packedEdges, 0.0, 1.0);\n"
	"}\n";

// AO denoise: XeGTAO_Denoise (edge-aware 3x3 blur, one pixel per invocation; the HLSL does two).
// sampler0 = AO working term (R) + packed edges (G); rcpFrame.zw = target size;
// aoParams.x = blur amount (DenoiseBlurBeta 1.2 on the final pass, /5 before), aoParams.w = final scale
// (XE_GTAO_OCCLUSION_TERM_SCALE on the final pass, 1 otherwise). Edges are passed through in G.
const char *s_PostAODenoise =
	"vec4 nlGtaoUnpackEdges(float packedVal)\n"
	"{\n"
	"  uint p = uint(packedVal * 255.5);\n"
	"  vec4 e;\n"
	"  e.x = float((p >> 6u) & 3u) / 3.0;\n"
	"  e.y = float((p >> 4u) & 3u) / 3.0;\n"
	"  e.z = float((p >> 2u) & 3u) / 3.0;\n"
	"  e.w = float(p & 3u) / 3.0;\n"
	"  return clamp(e, 0.0, 1.0);\n"
	"}\n"
	"vec4 nlAOFetch(ivec2 c, ivec2 sz) { return texelFetch(sampler0, clamp(c, ivec2(0), sz - 1), 0); }\n"
	"void main()\n"
	"{\n"
	"  ivec2 sz = ivec2(rcpFrame.zw);\n"
	"  ivec2 p = ivec2(gl_FragCoord.xy);\n"
	"  float blurAmount = aoParams.x;\n"
	"  float finalScale = aoParams.w;\n"
	"  const float diagWeight = 0.85 * 0.5;\n"
	"  vec4 c  = nlAOFetch(p, sz);\n"
	"  vec4 l  = nlAOFetch(p + ivec2(-1,  0), sz);\n"
	"  vec4 r  = nlAOFetch(p + ivec2( 1,  0), sz);\n"
	"  vec4 t  = nlAOFetch(p + ivec2( 0,  1), sz);\n"
	"  vec4 b  = nlAOFetch(p + ivec2( 0, -1), sz);\n"
	"  vec4 tl = nlAOFetch(p + ivec2(-1,  1), sz);\n"
	"  vec4 tr = nlAOFetch(p + ivec2( 1,  1), sz);\n"
	"  vec4 bl = nlAOFetch(p + ivec2(-1, -1), sz);\n"
	"  vec4 br = nlAOFetch(p + ivec2( 1, -1), sz);\n"
	"  vec4 eC = nlGtaoUnpackEdges(c.g);\n"
	"  vec4 eL = nlGtaoUnpackEdges(l.g);\n"
	"  vec4 eR = nlGtaoUnpackEdges(r.g);\n"
	"  vec4 eT = nlGtaoUnpackEdges(t.g);\n"
	"  vec4 eB = nlGtaoUnpackEdges(b.g);\n"
	// enforce edge symmetry with the neighbours
	"  eC *= vec4(eL.y, eR.x, eT.w, eB.z);\n"
	// small leak through 3-4 edge pixels (reduces aliasing)
	"  const float leakThreshold = 2.5;\n"
	"  const float leakStrength = 0.5;\n"
	"  float edginess = (clamp(4.0 - leakThreshold - dot(eC, vec4(1.0)), 0.0, 1.0) / (4.0 - leakThreshold)) * leakStrength;\n"
	"  eC = clamp(eC + edginess, 0.0, 1.0);\n"
	"  float wTL = diagWeight * (eC.x * eL.z + eC.z * eT.x);\n"
	"  float wTR = diagWeight * (eC.z * eT.y + eC.y * eR.z);\n"
	"  float wBL = diagWeight * (eC.w * eB.x + eC.x * eL.w);\n"
	"  float wBR = diagWeight * (eC.y * eR.w + eC.w * eB.y);\n"
	"  float sumWeight = blurAmount;\n"
	"  float sum = c.r * sumWeight;\n"
	"  sum += l.r * eC.x;  sumWeight += eC.x;\n"
	"  sum += r.r * eC.y;  sumWeight += eC.y;\n"
	"  sum += t.r * eC.z;  sumWeight += eC.z;\n"
	"  sum += b.r * eC.w;  sumWeight += eC.w;\n"
	"  sum += tl.r * wTL;  sumWeight += wTL;\n"
	"  sum += tr.r * wTR;  sumWeight += wTR;\n"
	"  sum += bl.r * wBL;  sumWeight += wBL;\n"
	"  sum += br.r * wBR;  sumWeight += wBR;\n"
	"  float ao = sum / sumWeight;\n"
	"  fragColor = vec4(clamp(ao * finalScale, 0.0, 1.0), c.g, 0.0, 1.0);\n"
	"}\n";

// AO composite (in place on the scene target, depth unbound): scene (sampler0) * mix(1, ao, strength),
// ao (sampler1, bilinear from half-res at tiers 1-2) floored so the result never drops below 40 %.
// aoParams.y = strength, aoParams.z > 0.5 = NL_POST_DEBUG raw AO view.
const char *s_PostAOComposite =
	"void main()\n"
	"{\n"
	"  vec4 s = texture(sampler0, texCoord0.xy);\n"
	"  float ao = texture(sampler1, texCoord0.xy).r;\n"
	"  if (aoParams.z > 0.5)\n"
	"  {\n"
	"    fragColor = vec4(vec3(ao), s.a);\n"
	"    return;\n"
	"  }\n"
	"  float f = max(mix(1.0, ao, aoParams.y), 0.4);\n"
	"  fragColor = vec4(s.rgb * f, s.a);\n"
	"}\n";

// ***************************************************************************
// NlPost UBO (std140): vec4 rcpFrame; vec4 projParams; vec4 aoParams; vec4 bloomParams; vec4 gradeParams;
// ***************************************************************************
struct CPostUBOOffsets
{
	sint RcpFrame;
	sint ProjParams;
	sint AOParams;
	sint BloomParams;
	sint GradeParams;
};
static CPostUBOOffsets s_PostUBOOffsets;
static NLMISC::CSmartPtr<CUniformBufferFormat> s_PostUBFormat;
static NLMISC::CSmartPtr<CUniformBuffer> s_PostUB;

// FXAA 3.11 tiers: Subpix, EdgeThreshold, EdgeThresholdMin
static const float s_FXAATiers[3][3] = {
	{ 0.0f, 0.0f, 0.0f },
	{ 0.5f, 0.25f, 0.0833f },
	{ 0.75f, 0.166f, 0.0833f },
};

} /* anonymous namespace */

class CVertexProgramPost : public CVertexProgram
{
public:
	struct CIdx
	{
		uint ModelViewProjection;
	};

	virtual void buildInfo() NL_OVERRIDE
	{
		CVertexProgram::buildInfo();
		m_Idx.ModelViewProjection = ~0u;
		if (source() && source()->Features.OnlyUBOs)
			return;
		m_Idx.ModelViewProjection = getUniformIndex("modelViewProjection");
	}

	const CIdx &idx() const { return m_Idx; }

private:
	CIdx m_Idx;
};

class CPixelProgramPost : public CPixelProgram
{
public:
	struct CIdx
	{
		uint RcpFrame;
		uint ProjParams;
		uint AOParams;
		uint BloomParams;
		uint GradeParams;
	};

	virtual void buildInfo() NL_OVERRIDE
	{
		CPixelProgram::buildInfo();
		m_Idx.RcpFrame = m_Idx.ProjParams = m_Idx.AOParams = m_Idx.BloomParams = m_Idx.GradeParams = ~0u;
		if (source() && source()->Features.OnlyUBOs)
			return;
		// Unused uniforms are optimised out by the GLSL compiler: ~0u indices are skipped by the driver
		m_Idx.RcpFrame = getUniformIndex("rcpFrame");
		m_Idx.ProjParams = getUniformIndex("projParams");
		m_Idx.AOParams = getUniformIndex("aoParams");
		m_Idx.BloomParams = getUniformIndex("bloomParams");
		m_Idx.GradeParams = getUniformIndex("gradeParams");
	}

	const CIdx &idx() const { return m_Idx; }

private:
	CIdx m_Idx;
};

} /* namespace NL3D */

#ifdef DEBUG_NEW
#define new DEBUG_NEW
#endif

namespace NL3D {

// ***************************************************************************
CPostStack::CPostParams::CPostParams()
{
	for (uint i = 0; i < 4; ++i)
		RcpFrame[i] = ProjParams[i] = AOParams[i] = BloomParams[i] = GradeParams[i] = 0.0f;
}

// ***************************************************************************
CPostStack::CPostStack(NL3D::UDriver *driver)
    : m_Driver(driver)
    , m_Supported(false)
    , m_VP(nullptr)
    , m_PPBright(nullptr)
    , m_PPDown(nullptr)
    , m_PPUp(nullptr)
    , m_PPComposite(nullptr)
    , m_PPGrade(nullptr)
    , m_PPSharpen(nullptr)
    , m_PPGTAO(nullptr)
    , m_PPAODenoise(nullptr)
    , m_PPAOComposite(nullptr)
    , m_FXAA(nullptr)
    , m_DebugAO(false)
    , m_AO(0)
    , m_AA(0)
    , m_Bloom(false)
    , m_BloomIntensity(0.35f)
    , m_RenderScale(100)
    , m_Sharpness(0.3f)
    , m_Grade(1.0f)
    , m_SceneTarget(nullptr)
    , m_Width(0)
    , m_Height(0)
{
	nldebug("3D: Create post stack");
	m_ProjParams[0] = m_ProjParams[1] = m_ProjParams[2] = m_ProjParams[3] = 0.0f;

	CDriverUser *dru = static_cast<CDriverUser *>(driver);
	NL3D::IDriver *drv = dru->getDriver();

	if (!(drv->supportBloomEffect() && drv->supportNonPowerOfTwoTextures()))
	{
		nldebug("3D: Post stack unsupported (no offscreen render targets)");
		return;
	}

	// NlPost UBO format (once)
	if (!s_PostUBFormat)
	{
		CUniformBufferFormat *fmt = new CUniformBufferFormat();
		fmt->Name = "NlPost";
		s_PostUBOOffsets.RcpFrame = fmt->push("rcpFrame", CUniformBufferFormat::FloatVec4);
		s_PostUBOOffsets.ProjParams = fmt->push("projParams", CUniformBufferFormat::FloatVec4);
		s_PostUBOOffsets.AOParams = fmt->push("aoParams", CUniformBufferFormat::FloatVec4);
		s_PostUBOOffsets.BloomParams = fmt->push("bloomParams", CUniformBufferFormat::FloatVec4);
		s_PostUBOOffsets.GradeParams = fmt->push("gradeParams", CUniformBufferFormat::FloatVec4);
		s_PostUBFormat = fmt;
		s_PostUB = new CUniformBuffer();
		s_PostUB->Format = *fmt;
	}

	// Pixel programs
	m_PPBright = createPixelProgram("Bright", s_PostBright);
	m_PPDown = createPixelProgram("Down", s_PostDown);
	m_PPUp = createPixelProgram("Up", s_PostUp);
	m_PPComposite = createPixelProgram("Composite", s_PostComposite);
	m_PPGrade = createPixelProgram("Grade", s_PostGrade);
	m_PPSharpen = createPixelProgram("Sharpen", s_PostSharpen);
	if (!(m_PPBright && m_PPDown && m_PPUp && m_PPComposite && m_PPGrade && m_PPSharpen))
	{
		nldebug("3D: Post stack unsupported (GL3 pixel programs not available on this driver)");
		release();
		return;
	}

	// AO programs (optional: the stage is skipped when the driver has no sampleable scene depth)
	if (drv->supportRenderTargetDepthTexture())
	{
		m_PPGTAO = createPixelProgram("GTAO", s_PostGTAO);
		m_PPAODenoise = createPixelProgram("AODenoise", s_PostAODenoise);
		m_PPAOComposite = createPixelProgram("AOComposite", s_PostAOComposite);
		if (!(m_PPGTAO && m_PPAODenoise && m_PPAOComposite))
		{
			nlwarning("3D: Post stack AO programs failed to compile; AO stage disabled");
			delete m_PPGTAO; m_PPGTAO = nullptr;
			delete m_PPAODenoise; m_PPAODenoise = nullptr;
			delete m_PPAOComposite; m_PPAOComposite = nullptr;
		}
	}

	// NL_POST_DEBUG=1: the AO composite outputs the raw AO term instead of the shaded scene
	{
		const char *dbg = getenv("NL_POST_DEBUG");
		m_DebugAO = dbg && atoi(dbg) > 0;
		if (m_DebugAO)
			nlinfo("3D: Post stack debug view enabled (NL_POST_DEBUG=%s): raw AO term", dbg);
	}

	// Vertex program
	{
		m_VP = new CVertexProgramPost();
		// glsl300esv — linked program path (NlModel UBO)
		{
			IProgram::CSource *source = new IProgram::CSource();
			source->Features.MaterialFlags = CProgramFeatures::TextureStages;
			source->Features.OnlyUBOs = true;
			source->Features.UsesObjectUBO = true;
			source->Profile = IProgram::glsl300esv;
			source->DisplayName = "glsl300esv/Post_VP/UBO";
			source->setSourcePtr(s_PostVPES);
			m_VP->addSource(source);
		}
		// glsl330v — SSO fallback
		{
			IProgram::CSource *source = new IProgram::CSource();
			source->Features.MaterialFlags = CProgramFeatures::TextureStages;
			source->Profile = IProgram::glsl330v;
			source->DisplayName = "glsl330v/Post_VP";
			source->setSourcePtr(s_PostVP330);
			m_VP->addSource(source);
		}
		if (!drv->compileVertexProgram(m_VP))
		{
			nlwarning("3D: No supported vertex program for the post stack");
			release();
			return;
		}
	}

	// FXAA stage (shared GLSL path, quality set per tier)
	m_FXAA = new CFXAA(driver);
	if (!m_FXAA->isSupported())
	{
		nlwarning("3D: FXAA stage unavailable for the post stack");
		delete m_FXAA;
		m_FXAA = nullptr;
	}

	// Materials and quad
	{
		m_Mat = m_Driver->createMaterial();
		m_Mat.initUnlit();
		m_Mat.setColor(CRGBA::White);
		m_Mat.setBlend(false);
		m_Mat.setAlphaTest(false);
		NL3D::CMaterial *mat = m_Mat.getObjectPtr();
		mat->setShader(NL3D::CMaterial::Normal);
		mat->setBlendFunc(CMaterial::one, CMaterial::zero);
		mat->setZWrite(false);
		mat->setZFunc(CMaterial::always);
		mat->setDoubleSided(true);

		m_MatAdd = m_Driver->createMaterial();
		m_MatAdd.initUnlit();
		m_MatAdd.setColor(CRGBA::White);
		m_MatAdd.setBlend(true);
		m_MatAdd.setAlphaTest(false);
		NL3D::CMaterial *matAdd = m_MatAdd.getObjectPtr();
		matAdd->setShader(NL3D::CMaterial::Normal);
		matAdd->setBlendFunc(CMaterial::one, CMaterial::one);
		matAdd->setZWrite(false);
		matAdd->setZFunc(CMaterial::always);
		matAdd->setDoubleSided(true);

		m_QuadUV.V0 = CVector(0.f, 0.f, 0.5f);
		m_QuadUV.V1 = CVector(1.f, 0.f, 0.5f);
		m_QuadUV.V2 = CVector(1.f, 1.f, 0.5f);
		m_QuadUV.V3 = CVector(0.f, 1.f, 0.5f);
		m_QuadUV.Uv0 = CUV(0.f, 0.f);
		m_QuadUV.Uv1 = CUV(1.f, 0.f);
		m_QuadUV.Uv2 = CUV(1.f, 1.f);
		m_QuadUV.Uv3 = CUV(0.f, 1.f);
	}

	m_Supported = true;
	nldebug("3D: Post stack available (scene depth texture: %s)", drv->supportRenderTargetDepthTexture() ? "yes" : "no");
}

// ***************************************************************************
CPostStack::~CPostStack()
{
	nldebug("3D: Destroy post stack");
	release();
	m_Driver = nullptr;
}

// ***************************************************************************
void CPostStack::release()
{
	m_Supported = false;
	if (!m_Mat.empty())
		m_Driver->deleteMaterial(m_Mat);
	if (!m_MatAdd.empty())
		m_Driver->deleteMaterial(m_MatAdd);
	delete m_FXAA; m_FXAA = nullptr;
	delete m_VP; m_VP = nullptr;
	delete m_PPBright; m_PPBright = nullptr;
	delete m_PPDown; m_PPDown = nullptr;
	delete m_PPUp; m_PPUp = nullptr;
	delete m_PPComposite; m_PPComposite = nullptr;
	delete m_PPGrade; m_PPGrade = nullptr;
	delete m_PPSharpen; m_PPSharpen = nullptr;
	delete m_PPGTAO; m_PPGTAO = nullptr;
	delete m_PPAODenoise; m_PPAODenoise = nullptr;
	delete m_PPAOComposite; m_PPAOComposite = nullptr;
}

// ***************************************************************************
CPixelProgram *CPostStack::createPixelProgram(const char *name, const char *body)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	NL3D::IDriver *drv = dru->getDriver();

	CPixelProgramPost *pp = new CPixelProgramPost();
	// glsl300esf — linked program path, NlPost UBO
	{
		IProgram::CSource *source = new IProgram::CSource();
		source->Features.MaterialFlags = CProgramFeatures::TextureStages;
		source->Features.OnlyUBOs = true;
		source->Features.NoBuiltinUniforms = true;
		source->UniformBufferFormats[UBBindingPixelProgram] = s_PostUBFormat;
		source->Profile = IProgram::glsl300esf;
		source->DisplayName = std::string("glsl300esf/Post_") + name + "/UBO";
		source->setSource(std::string(s_PostPPHeaderES) + s_PostCommon + body);
		pp->addSource(source);
	}
	// glsl330f — SSO fallback, plain uniforms
	{
		IProgram::CSource *source = new IProgram::CSource();
		source->Features.MaterialFlags = CProgramFeatures::TextureStages;
		source->Features.NoBuiltinUniforms = true;
		source->Profile = IProgram::glsl330f;
		source->DisplayName = std::string("glsl330f/Post_") + name;
		source->setSource(std::string(s_PostPPHeader330) + s_PostCommon + body);
		pp->addSource(source);
	}
	if (!drv->compilePixelProgram(pp))
	{
		nlwarning("3D: No supported pixel program for post pass '%s'", name);
		delete pp;
		return nullptr;
	}
	return pp;
}

// ***************************************************************************
void CPostStack::setQuality(uint ao, uint aa, bool bloom, float bloomIntensity, uint renderScale, float sharpness, float grade)
{
	m_AO = std::min(ao, 3u);
	m_AA = std::min(aa, 2u);
	m_Bloom = bloom;
	m_BloomIntensity = std::min(std::max(bloomIntensity, 0.0f), 1.0f);
	m_RenderScale = std::min(std::max(renderScale, 50u), 200u);
	m_Sharpness = std::min(std::max(sharpness, 0.0f), 1.0f);
	m_Grade = std::min(std::max(grade, 0.0f), 1.0f);
}

// ***************************************************************************
bool CPostStack::anyStageEnabled() const
{
	if (!m_Supported)
		return false;
	return m_AO > 0 || m_AA > 0 || m_Bloom || m_RenderScale != 100;
}

// ***************************************************************************
void CPostStack::getScaledSize(uint32 windowW, uint32 windowH, uint32 &w, uint32 &h) const
{
	if (!m_Supported || m_RenderScale == 100)
	{
		w = windowW;
		h = windowH;
		return;
	}
	w = std::max(1u, (uint32)((float)windowW * (float)m_RenderScale / 100.0f + 0.5f));
	h = std::max(1u, (uint32)((float)windowH * (float)m_RenderScale / 100.0f + 0.5f));
}

// ***************************************************************************
void CPostStack::setParams(CPixelProgram *pp, const CPostParams &p)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();

	bool ppUBO = pp->source() && pp->source()->Features.OnlyUBOs;
	if (ppUBO)
	{
		s_PostUB->lock();
		s_PostUB->set(s_PostUBOOffsets.RcpFrame, p.RcpFrame[0], p.RcpFrame[1], p.RcpFrame[2], p.RcpFrame[3]);
		s_PostUB->set(s_PostUBOOffsets.ProjParams, p.ProjParams[0], p.ProjParams[1], p.ProjParams[2], p.ProjParams[3]);
		s_PostUB->set(s_PostUBOOffsets.AOParams, p.AOParams[0], p.AOParams[1], p.AOParams[2], p.AOParams[3]);
		s_PostUB->set(s_PostUBOOffsets.BloomParams, p.BloomParams[0], p.BloomParams[1], p.BloomParams[2], p.BloomParams[3]);
		s_PostUB->set(s_PostUBOOffsets.GradeParams, p.GradeParams[0], p.GradeParams[1], p.GradeParams[2], p.GradeParams[3]);
		s_PostUB->unlock();
		drv->bindUniformBuffer(UBBindingPixelProgram, s_PostUB);
	}
	else
	{
		const CPixelProgramPost::CIdx &idx = static_cast<CPixelProgramPost *>(pp)->idx();
		if (idx.RcpFrame != ~0u) drv->setUniform4f(IDriver::PixelProgram, idx.RcpFrame, p.RcpFrame[0], p.RcpFrame[1], p.RcpFrame[2], p.RcpFrame[3]);
		if (idx.ProjParams != ~0u) drv->setUniform4f(IDriver::PixelProgram, idx.ProjParams, p.ProjParams[0], p.ProjParams[1], p.ProjParams[2], p.ProjParams[3]);
		if (idx.AOParams != ~0u) drv->setUniform4f(IDriver::PixelProgram, idx.AOParams, p.AOParams[0], p.AOParams[1], p.AOParams[2], p.AOParams[3]);
		if (idx.BloomParams != ~0u) drv->setUniform4f(IDriver::PixelProgram, idx.BloomParams, p.BloomParams[0], p.BloomParams[1], p.BloomParams[2], p.BloomParams[3]);
		if (idx.GradeParams != ~0u) drv->setUniform4f(IDriver::PixelProgram, idx.GradeParams, p.GradeParams[0], p.GradeParams[1], p.GradeParams[2], p.GradeParams[3]);
	}
}

// ***************************************************************************
void CPostStack::drawPass(CPixelProgram *pp, ITexture *tex0, ITexture *tex1, UMaterial &mat, const CPostParams &p)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();

	m_Driver->setMatrixMode2D11();

	bool vpok = drv->activeVertexProgram(m_VP);
	nlassert(vpok);
	bool ppok = drv->activePixelProgram(pp);
	nlassert(ppok);

	bool ppUBO = pp->source() && pp->source()->Features.OnlyUBOs;
	bool vpUBO = m_VP->source() && m_VP->source()->Features.OnlyUBOs;

	setParams(pp, p);
	if (!vpUBO)
	{
		drv->setUniformMatrix(IDriver::VertexProgram, static_cast<CVertexProgramPost *>(m_VP)->idx().ModelViewProjection, IDriver::ModelViewProjection, IDriver::Identity);
	}

	CMaterial *m = mat.getObjectPtr();
	m->setTexture(0, tex0);
	m->setTexture(1, tex1);
	m_Driver->drawQuad(m_QuadUV, mat);
	m->setTexture(0, nullptr);
	m->setTexture(1, nullptr);

	drv->activeVertexProgram(nullptr);
	drv->activePixelProgram(nullptr);
	if (ppUBO) drv->bindUniformBuffer(UBBindingPixelProgram, nullptr);
}

// ***************************************************************************
void CPostStack::restoreSceneTarget()
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();
	if (drv->getRenderTarget() != m_SceneTarget)
	{
		// Always go through the backbuffer so the driver's saved viewport stays the caller's
		CTextureUser texNull;
		dru->setRenderTarget(texNull);
		drv->setRenderTarget(m_SceneTarget);
	}
}

// ***************************************************************************
void CPostStack::runInPlace(CPixelProgram *pp, ITexture *tex1, const CPostParams &p)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();

	ITexture *renderTarget = m_SceneTarget;
	bool needsDepthStencil = static_cast<CTextureOffscreen *>(renderTarget)->needsDepthStencil();

	// Same-kind scratch (CFXAA pattern): swap GL handles so the scratch holds the current image,
	// then render back into the scene target object. Same kind keeps the shared depth-stencil
	// attachment consistent across the swap.
	CTextureUser *scratch = m_Driver->getRenderTargetManager().getRenderTarget(m_Width, m_Height, needsDepthStencil);
	nlassert(scratch);

	CTextureUser texNull;
	dru->setRenderTarget(texNull);
	drv->swapTextureHandle(*renderTarget, *scratch->getITexture());
	drv->setRenderTarget(renderTarget);

	drawPass(pp, scratch->getITexture(), tex1, m_Mat, p);

	m_Driver->getRenderTargetManager().recycleRenderTarget(scratch);
}

// ***************************************************************************
void CPostStack::runTo(CPixelProgram *pp, ITexture *tex0, ITexture *tex1, CTextureUser *dest, bool additive, const CPostParams &p)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);

	CTextureUser texNull;
	dru->setRenderTarget(texNull);
	dru->setRenderTarget(*dest);

	drawPass(pp, tex0, tex1, additive ? m_MatAdd : m_Mat, p);
}

// ***************************************************************************
bool CPostStack::bindSceneDepth(bool bind)
{
	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();
	if (!drv->supportRenderTargetDepthTexture())
		return false;
	return drv->bindRenderTargetDepthTexture(bind ? m_SceneTarget : nullptr);
}

// ***************************************************************************
void CPostStack::apply(NL3D::UScene *scene)
{
	if (!m_Supported)
		return;

	CDriverUser *dru = static_cast<CDriverUser *>(m_Driver);
	IDriver *drv = dru->getDriver();

	ITexture *sceneTarget = drv->getRenderTarget();
	nlassert(sceneTarget);
	nlassert(sceneTarget->isOffscreenTexture());
	nlassert(sceneTarget->getUploadFormat() == ITexture::Auto);
	m_SceneTarget = sceneTarget;
	m_Width = sceneTarget->getWidth();
	m_Height = sceneTarget->getHeight();

	// backup
	bool fogEnabled = m_Driver->fogEnabled();
	m_Driver->enableFog(false);

	// Camera-derived depth linearisation params (consumed by the AO pass, part B)
	m_ProjParams[0] = m_ProjParams[1] = m_ProjParams[2] = m_ProjParams[3] = 0.0f;
	if (scene)
	{
		UCamera cam = scene->getCam();
		const CFrustum &f = cam.getFrustum();
		m_ProjParams[0] = f.Near;
		m_ProjParams[1] = f.Far;
		float dw = f.Right - f.Left;
		float dh = f.Top - f.Bottom;
		m_ProjParams[2] = (dw != 0.0f) ? (2.0f * f.Near / dw) : 1.0f;
		m_ProjParams[3] = (dh != 0.0f) ? (2.0f * f.Near / dh) : 1.0f;
	}

	// Pass chain (design order)
	if (m_AO > 0)
		applyAO();
	if (m_Bloom)
		applyBloom();
	if (m_Grade > 0.0f)
		applyGrade();
	if (m_AA > 0)
		applyFXAA();
	if (m_RenderScale < 100 && m_Sharpness > 0.0f)
		applySharpen();

	// Leave the scene target current for endDefaultRenderTarget()
	restoreSceneTarget();

	// restore
	m_Driver->enableFog(fogEnabled);
	m_SceneTarget = nullptr;
}

// ***************************************************************************
void CPostStack::applyAO()
{
	// GTAO (XeGTAO port). Tiers: 1 = half-res 2x2, 1 denoise; 2 = half-res 3x3, 1 denoise;
	// 3 = full-res 4x4, 2 denoise passes. Half-res tiers are upsampled bilinearly in the composite
	// (the AO targets are Linear-filtered by CRenderTargetManager).
	if (!(m_PPGTAO && m_PPAODenoise && m_PPAOComposite))
		return;
	if (m_ProjParams[1] <= m_ProjParams[0])
		return; // no camera (apply(NULL)): depth cannot be linearised
	if (!bindSceneDepth(true))
	{
		static bool warned = false;
		if (!warned)
		{
			nlwarning("3D: Post stack AO skipped: scene depth texture not available");
			warned = true;
		}
		return;
	}

	CRenderTargetManager &rtm = m_Driver->getRenderTargetManager();

	static const float s_AOSlices[4] = { 0.0f, 2.0f, 3.0f, 4.0f };
	static const float s_AOSteps[4] = { 0.0f, 2.0f, 3.0f, 4.0f };
	const float aoRadius = 1.5f;   // effective world radius [m] (XeGTAO Radius x RadiusMultiplier folded in)
	const float aoStrength = 0.6f; // composite mix; the composite also floors the result at 40 %
	const float denoiseBlurBeta = 1.2f; // XeGTAO DenoiseBlurBeta for DenoisePasses > 0
	const float occlusionTermScale = 1.5f; // XE_GTAO_OCCLUSION_TERM_SCALE

	uint tier = std::min(m_AO, 3u);
	bool fullRes = (tier >= 3);
	uint aw = fullRes ? m_Width : std::max(1u, (m_Width + 1) / 2);
	uint ah = fullRes ? m_Height : std::max(1u, (m_Height + 1) / 2);
	uint denoisePasses = fullRes ? 2 : 1;

	// Colour-only targets (no depth attachment): the scene depth texture is sampled while they are bound
	CTextureUser *aoA = rtm.getRenderTarget(aw, ah, false);
	CTextureUser *aoB = rtm.getRenderTarget(aw, ah, false);
	nlassert(aoA && aoB);

	CPostParams p;
	p.RcpFrame[0] = 1.0f / (float)aw;
	p.RcpFrame[1] = 1.0f / (float)ah;
	p.RcpFrame[2] = (float)aw;
	p.RcpFrame[3] = (float)ah;
	for (uint i = 0; i < 4; ++i)
		p.ProjParams[i] = m_ProjParams[i];

	// Main pass: scene depth -> AO working term + edges (sampler0 unused, bound for a complete material)
	p.AOParams[0] = aoRadius;
	p.AOParams[1] = fullRes ? 1.0f : 2.0f; // AO texel / depth texel ratio
	p.AOParams[2] = s_AOSlices[tier];
	p.AOParams[3] = s_AOSteps[tier];
	runTo(m_PPGTAO, m_SceneTarget, nullptr, aoA, false, p);

	// Edge-aware denoise, ping-pong; the last pass scales the working term back to visibility
	CTextureUser *src = aoA;
	CTextureUser *dst = aoB;
	for (uint i = 0; i < denoisePasses; ++i)
	{
		bool finalApply = (i + 1 == denoisePasses);
		p.AOParams[0] = finalApply ? denoiseBlurBeta : denoiseBlurBeta / 5.0f;
		p.AOParams[1] = 0.0f;
		p.AOParams[2] = 0.0f;
		p.AOParams[3] = finalApply ? occlusionTermScale : 1.0f;
		runTo(m_PPAODenoise, src->getITexture(), nullptr, dst, false, p);
		std::swap(src, dst);
	}

	// Composite in place on the scene target: its depth is attached again, so unbind the sampler first
	bindSceneDepth(false);
	p.RcpFrame[0] = 1.0f / (float)m_Width;
	p.RcpFrame[1] = 1.0f / (float)m_Height;
	p.RcpFrame[2] = (float)m_Width;
	p.RcpFrame[3] = (float)m_Height;
	p.AOParams[0] = aoRadius;
	p.AOParams[1] = aoStrength;
	p.AOParams[2] = m_DebugAO ? 1.0f : 0.0f;
	p.AOParams[3] = 0.0f;
	runInPlace(m_PPAOComposite, src->getITexture(), p);

	rtm.recycleRenderTarget(aoA);
	rtm.recycleRenderTarget(aoB);
}

// ***************************************************************************
void CPostStack::applyBloom()
{
	CRenderTargetManager &rtm = m_Driver->getRenderTargetManager();

	// Half-res chain, 4 levels, colour-only
	const uint numLevels = 4;
	CTextureUser *levels[numLevels];
	uint lw[numLevels], lh[numLevels];
	uint cw = m_Width, ch = m_Height;
	for (uint i = 0; i < numLevels; ++i)
	{
		cw = std::max(1u, cw / 2);
		ch = std::max(1u, ch / 2);
		lw[i] = cw;
		lh[i] = ch;
		levels[i] = rtm.getRenderTarget(cw, ch, false);
		nlassert(levels[i]);
	}

	CPostParams p;
	p.BloomParams[0] = 0.75f; // luma threshold
	p.BloomParams[1] = 0.1f;  // soft knee
	p.BloomParams[2] = m_BloomIntensity;

	// Bright pass: scene -> level 0
	p.RcpFrame[0] = 1.0f / (float)m_Width;
	p.RcpFrame[1] = 1.0f / (float)m_Height;
	p.RcpFrame[2] = (float)m_Width;
	p.RcpFrame[3] = (float)m_Height;
	runTo(m_PPBright, m_SceneTarget, nullptr, levels[0], false, p);

	// Downsample chain
	for (uint i = 1; i < numLevels; ++i)
	{
		p.RcpFrame[0] = 1.0f / (float)lw[i - 1];
		p.RcpFrame[1] = 1.0f / (float)lh[i - 1];
		p.RcpFrame[2] = (float)lw[i - 1];
		p.RcpFrame[3] = (float)lh[i - 1];
		runTo(m_PPDown, levels[i - 1]->getITexture(), nullptr, levels[i], false, p);
	}

	// Tent upsample, additive into the next larger level
	for (uint i = numLevels - 1; i >= 1; --i)
	{
		p.RcpFrame[0] = 1.0f / (float)lw[i];
		p.RcpFrame[1] = 1.0f / (float)lh[i];
		p.RcpFrame[2] = (float)lw[i];
		p.RcpFrame[3] = (float)lh[i];
		runTo(m_PPUp, levels[i]->getITexture(), nullptr, levels[i - 1], true, p);
	}

	// Composite in place on the scene target
	p.RcpFrame[0] = 1.0f / (float)m_Width;
	p.RcpFrame[1] = 1.0f / (float)m_Height;
	p.RcpFrame[2] = (float)m_Width;
	p.RcpFrame[3] = (float)m_Height;
	runInPlace(m_PPComposite, levels[0]->getITexture(), p);

	for (uint i = 0; i < numLevels; ++i)
		rtm.recycleRenderTarget(levels[i]);
}

// ***************************************************************************
void CPostStack::applyGrade()
{
	CPostParams p;
	p.RcpFrame[0] = 1.0f / (float)m_Width;
	p.RcpFrame[1] = 1.0f / (float)m_Height;
	p.RcpFrame[2] = (float)m_Width;
	p.RcpFrame[3] = (float)m_Height;
	p.GradeParams[0] = m_Grade; // strength
	p.GradeParams[1] = 0.3f;    // S-curve amount (mild)
	p.GradeParams[2] = 1.08f;   // saturation
	runInPlace(m_PPGrade, nullptr, p);
}

// ***************************************************************************
void CPostStack::applyFXAA()
{
	if (!m_FXAA)
		return;
	const float *tier = s_FXAATiers[std::min(m_AA, 2u)];
	m_FXAA->setQuality(tier[0], tier[1], tier[2]);
	// CFXAA works in place on the driver's current render target
	restoreSceneTarget();
	m_FXAA->applyEffect();
}

// ***************************************************************************
void CPostStack::applySharpen()
{
	CPostParams p;
	p.RcpFrame[0] = 1.0f / (float)m_Width;
	p.RcpFrame[1] = 1.0f / (float)m_Height;
	p.RcpFrame[2] = (float)m_Width;
	p.RcpFrame[3] = (float)m_Height;
	p.GradeParams[3] = m_Sharpness;
	runInPlace(m_PPSharpen, nullptr, p);
}

} /* namespace NL3D */

/* end of file */
