#include "ExpectedPathLength.t.hh"

#include <math.h>

#include "GridBins.hh"
#include "GPUTiming.hh"
#include "MonteRayDefinitions.hh"
#include "GPUAtomicAdd.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRayMaterialList.hh"

namespace MonteRay{

CUDA_CALLABLE_MEMBER
gpuTallyType_t
attenuateRayTraceOnly(const MonteRayMaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const HashLookup* pHash,
        unsigned HashBin,
        unsigned cell,
        gpuFloatType_t distance,
        gpuFloatType_t energy,
        gpuTallyType_t enteringFraction,
        ParticleType_t particleType)
{
    gpuTallyType_t totalXS = 0.0;
    unsigned numMaterials = getNumMats( pMatProps, cell);
    for( unsigned i=0; i<numMaterials; ++i ) {

        unsigned matID = getMatID(pMatProps, cell, i);
        gpuFloatType_t density = getDensity(pMatProps, cell, i );
        if( density > 1e-5 ) {
            //unsigned materialIndex = materialIDtoIndex(pMatList, matID);
            if( particleType == neutron ) {
                totalXS +=  getTotalXS( pMatList, matID, pHash, HashBin, energy, density);
            } else {
                totalXS +=  getTotalXS( pMatList, matID, energy, density);
            }
        }
    }

    gpuTallyType_t attenuation = 1.0;

    if( totalXS > 1e-5 ) {
        attenuation = exp( - totalXS*distance );
    }
    return enteringFraction * attenuation;

}

CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment( const MonteRayMaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const gpuFloatType_t* materialXS,
        gpuTallyType_t* tally,
        unsigned cell,
        gpuRayFloat_t distance,
        gpuFloatType_t energy,
        gpuFloatType_t weight,
        gpuTallyType_t opticalPathLength ) {

#ifdef DEBUG
    const bool debug = false;
#endif

    typedef gpuTallyType_t xs_t;
    typedef gpuTallyType_t attenuation_t;
    typedef gpuTallyType_t score_t;

    xs_t totalXS = 0.0;
    unsigned numMaterials = getNumMats( pMatProps, cell);

#ifdef DEBUG
    if( debug ) {
        printf("GPU::tallyCellSegment:: cell=%d, numMaterials=%d\n", cell, numMaterials);
    }
#endif

    for( unsigned i=0; i<numMaterials; ++i ) {

        unsigned matID = getMatID(pMatProps, cell, i);
        gpuFloatType_t density = getDensity(pMatProps, cell, i );
        if( density > 1e-5 ) {
            totalXS +=   materialXS[matID]*density;
        }
        //		if( debug ) {
        //			printf("GPU::tallyCellSegment::       material=%d, density=%f, xs=%f, totalxs=%f\n", i, density, xs, totalXS);
        //		}
    }

    attenuation_t attenuation = 1.0;
    score_t score = distance;
    gpuTallyType_t cellOpticalPathLength = totalXS*distance;

    if( totalXS >  1e-5 ) {
        attenuation =  exp( - cellOpticalPathLength );
        score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
    }
    score *= exp( -opticalPathLength ) * weight;

    gpu_atomicAdd( &tally[cell], score);

#ifdef DEBUG
    if( debug ) {
        printf("GPU::tallyCellSegment:: total score=%f\n", tally[cell] );
    }
#endif

    return cellOpticalPathLength;
}

}


