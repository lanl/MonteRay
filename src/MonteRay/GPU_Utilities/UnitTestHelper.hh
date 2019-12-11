#define GPU_CHECK_RESULT(EXPR) if (not *testVal){  printf("test fail at line %d in file %s \n", __LINE__, __FILE__); return; }
#define GPU_CHECK_CLOSE(VAL1, VAL2, DIFF) *testVal = *testVal && ((VAL1 - VAL2) < DIFF); GPU_CHECK_RESULT(*testVal)
#define GPU_CHECK_EQUAL(VAL1, VAL2) *testVal = *testVal && (VAL1 == VAL2); GPU_CHECK_RESULT(*testVal)
#define GPU_CHECK(EXPR) *testVal = *testVal && (EXPR); GPU_CHECK_RESULT(*testVal); 
