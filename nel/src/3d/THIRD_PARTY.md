# Third-party code in nel/src/3d

## XeGTAO (ambient occlusion)

`post_stack.cpp` — the `s_PostGTAO` and `s_PostAODenoise` GLSL sources are a fragment-shader port of the
main pass and the denoise pass of XeGTAO (Intel's implementation of "Practical Real-Time Strategies for
Accurate Indirect Occlusion", Jimenez et al. 2016), rewritten from `XeGTAO.hlsli` / `XeGTAO.h` for the
NeL post-processing stack: no depth MIP prefilter, in-place normal reconstruction, no bent normals, static
interleaved-gradient noise, one pixel per invocation in the denoise.

Source: https://github.com/GameTechDev/XeGTAO

```
MIT License

Copyright (C) 2016-2021, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
