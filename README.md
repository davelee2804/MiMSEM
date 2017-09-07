# MiMSEM
Mixed Mimetic Spectral Element Model 

This reposotory houses a sandbox test code for the mixed mimetic spectral element model

Equation set: Rotating shallow water
Domain: 2D doubly periodic
Language: Python

The model demonstrates conservation of:
- volume (exact, pointwise) 
- vorticity (weak form)
- energy (weak form, to truncation error in time)
- potential enstrophy (weak form, to truncation error in time, subject to exact spatial integration)
