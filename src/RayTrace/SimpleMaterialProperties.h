#ifndef SIMPLEMATERIALPROPERTIES_HH_
#define SIMPLEMATERIALPROPERTIES_HH_

#include "gpuGlobal.h"

namespace MonteRay{

#define NMAX_MATERIALS 3

struct SimpleCellProperties {
    unsigned numMats;
    gpuFloatType_t density[NMAX_MATERIALS];
    unsigned matID[NMAX_MATERIALS];
};

struct SimpleMaterialProperties {
    unsigned numCells;
    struct SimpleCellProperties* props;
};

void ctor(SimpleMaterialProperties*, unsigned num );
void dtor(SimpleMaterialProperties* );
void copy(struct SimpleMaterialProperties* pCopy, struct SimpleMaterialProperties* pOrig);
void copy(SimpleCellProperties& theCopy, const SimpleCellProperties& theOrig);
void copy(SimpleCellProperties* pCopy, const SimpleCellProperties* pOrig);


#ifdef CUDA
void cudaCtor(struct SimpleMaterialProperties*,struct SimpleMaterialProperties*);
void cudaDtor(struct SimpleMaterialProperties*);
#endif

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumCells(struct SimpleMaterialProperties* ptr );

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumMats(struct SimpleMaterialProperties* ptr, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getDensity(struct SimpleMaterialProperties* ptr, unsigned cellNum, unsigned matNum );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMatID(struct SimpleMaterialProperties* ptr, unsigned cellNum, unsigned matNum );

#ifdef CUDA
__global__ void kernelGetNumCells(SimpleMaterialProperties* mp, unsigned* results );
#endif

#ifdef CUDA
__global__ void kernelSumMatDensity(SimpleMaterialProperties* mp, unsigned matIndex, gpuFloatType_t* results );
#endif

void addDensityAndID(struct SimpleMaterialProperties* ptr, unsigned cellNum, gpuFloatType_t density, unsigned matID );

class SimpleMaterialPropertiesHost {
public:

    SimpleMaterialPropertiesHost(unsigned numCells);

    ~SimpleMaterialPropertiesHost();

    void copyToGPU(void);

    unsigned getNumCells(void) const {
        return MonteRay::getNumCells( ptr );
    }

    unsigned launchGetNumCells(void) const;
    unsigned launchSumMatDensity(unsigned matID) const;

    unsigned getNumMats(unsigned i) const {
        return MonteRay::getNumMats( ptr, i);
    }

    unsigned getMaxNumMats(void) const {
        return NMAX_MATERIALS;
    }

    gpuFloatType_t getDensity(unsigned cell, unsigned matNum) const {
        return MonteRay::getDensity( ptr, cell, matNum );
    }

    unsigned getMatID(unsigned cell, unsigned matNum) const {
         return MonteRay::getMatID( ptr, cell, matNum );
     }

#ifndef CUDA
    void loadFromLnk3dnt(const std::string& filename );
#endif

    template<typename T>
    void loadFromReadLnk3dnt(const T& reader ){
        unsigned numCells = reader.getNumTotalCells();

        dtor( ptr );
        ctor( ptr, numCells );

        unsigned numMats = reader.MaxMaterialsPerCell();
        //std::cout << "Debug: numMats=" << numMats << std::endl;

        for( unsigned cell=0; cell < numCells; ++cell) {
            for( unsigned matNum=0; matNum < numMats; ++matNum ) {
                gpuFloatType_t density = reader.getDensity(cell, matNum);
                int matID = reader.getMatID(cell, matNum);

                matID -= 2;  // decrement partisn id numbers as they start with 1, and 1 is the ghost mat
                if( matID < 0 ) {
                    // partisn ghost id
                    continue;
                }

                addDensityAndID(cell, density, matID);
            }
        }
    }

    gpuFloatType_t sumMatDensity( unsigned matIndex) const;

    struct SimpleMaterialProperties* getPtr(void) { return ptr; }

    void addDensityAndID(unsigned cellNum, gpuFloatType_t density, unsigned matID ) {
    	MonteRay::addDensityAndID(ptr, cellNum, density, matID );
    }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void write( const std::string& filename ) const;
    void read( const std::string& filename );

    void write(std::ostream& outfile, const SimpleCellProperties& cellProp) const;
    void read(std::istream& outfile, SimpleCellProperties& cellProp);

private:
    SimpleMaterialProperties* ptr;
    SimpleMaterialProperties* temp;
    bool cudaCopyMade;

public:
    SimpleMaterialProperties* ptr_device;

};

}

#endif /* SIMPLEMATERIALPROPERTIES_HH_ */
