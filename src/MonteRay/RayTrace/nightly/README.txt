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

---------------------
Dec. 17, 2019
Feature/MovingMaterialCorrection Branch - refactored geometry, managed memory - no moving material corrections

Criticality Accident, maxdiff = 0.2206
gpuTime  = 16.2802
cpuTime  = 0.00198706 
wallTime = 16.2803

PWR Assembly, maxdiff = 0.337626
gpuTime  = 9.81557
cpuTime  = 0.000490334 
wallTime = 9.81564

Zeus2 Cylindrical, maxdiff = 0.0332647
Debug: total gpuTime  = 0.632041
Debug: total cpuTime  = 0.000402548
Debug: total wallTime = 0.632095

---------------------
Dec. 19, 2019
Feature/MovingMaterialCorrection Branch - refactored geometry, managed memory, cell-by-cell moving material correction (0 velocities)

Criticality Accident, maxdiff = 91.0 (large diffs, relatively small magnitude, some more concerning numbers need investigation)
gpuTime  = 13.9273

PWR Assembly, maxdiff = 0.43
gpuTime  = 5.71791

Zeus2 Cylindrical, maxdiff = 0.006
gpuTime  = 0.57




---------------------



---------------------
