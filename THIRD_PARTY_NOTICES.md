# Third-party notices

## SkinTokens

This repository does not redistribute SkinTokens. On first use, its setup clones SkinTokens from
https://github.com/VAST-AI-Research/SkinTokens and downloads the checkpoints from
https://huggingface.co/VAST-AI/SkinTokens on the user's machine.

`skintokens_runtime.py` contains short excerpts of SkinTokens source code. They locate the lines it patches in
the user's local copy: the PyTorch SDPA attention fallback and the progress markers. SkinTokens is distributed
under the MIT License:

```
MIT License

Copyright (c) 2025 VAST-AI-Research

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
