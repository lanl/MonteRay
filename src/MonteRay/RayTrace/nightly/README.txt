Testing notes

---------------------
Sept. 16, 2019
Testing prior to changes to use managed memory.

Hardware:
Darwin cn114, Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz, GeForce GTX TITAN X (GM200)

Criticality Accident:
gpuTime = 17.2482
cpuTime = 0.376569
wallTime = 17.2484

PWR Assembly:
gpuTime = 10.6651
cpuTime = 0.165921
wallTime = 10.665

Zeus2 Cylindrical:
Debug: total gpuTime  = 1.22129
Debug: total cpuTime  = 0.0871151
Debug: total wallTime = 1.2213

---------------------
Dec. 09, 2019
Master Branch w/ Shadow Memory, prior to Managed Memory changes

Hardware:
Benchmark, Intel(R) Xeon(R) CPU E5-2665,  GeForce GTX 1080 (Pascal), CUDA 10.2

Criticality Accident:
gpuTime  = 24.213
cpuTime  = 2.57113 
wallTime = 24.2131

PWR Assembly:
gpuTime  = 12.6281
cpuTime  = 0.244983
wallTime = 12.6282

Zeus2 Cylindrical:
Debug: total gpuTime  = 1.37924
Debug: total cpuTime  = 0.179672
Debug: total wallTime = 1.37927

Feature/MovingMaterialCorrection Branch - refactored geometry, managed memory - no moving material corrections

Criticality Accident:
gpuTime  = 17.9427 - up to 18.8 depending on run - maxdiff = 0.2206
cpuTime  = 1.48945
wallTime = 17.9428

PWR Assembly:
gpuTime  = 10.0843
cpuTime  = 0.239491
wallTime = 10.0844

Zeus2 Cylindrical:
Debug: total gpuTime  = 0.79552 - up to 0.82 depending on run
Debug: total cpuTime  = 0.175094
Debug: total wallTime = 0.795558




---------------------



---------------------
