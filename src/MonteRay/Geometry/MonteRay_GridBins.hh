#ifndef MonteRay_GridBins_HH_
#define MonteRay_GridBins_HH_

#ifndef __CUDA_ARCH__
#include <vector>
#endif

#include "MonteRayTypes.hh"
#include "MonteRayCopyMemory.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay {

class MonteRay_GridBins : public CopyMemoryBase<MonteRay_GridBins>{
public:
    using Base = CopyMemoryBase<MonteRay_GridBins> ;
    //typedef MonteRay_GridBins[3] gridInfoArray_t;
    typedef Vector3D<gpuRayFloat_t> Position_t;
    typedef Vector3D<gpuRayFloat_t> Direction_t;

    enum coordinate_t{ LINEAR, RADIAL };

    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins(void) {
        init();
    }


    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins( gpuRayFloat_t min, gpuRayFloat_t max, unsigned nBins) : CopyMemoryBase<MonteRay_GridBins>()  {
        init();
        initialize( min, max, nBins );
    }

    template<typename T>
    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins( const std::vector<T>& bins ) {
        init();
        initialize( bins );
    }

    MonteRay_GridBins&
    operator=( MonteRay_GridBins& rhs );

    //MonteRay_GridBins(const MonteRay_GridBins&);

    //MonteRay_GridBins& operator=( const MonteRay_GridBins& rhs );

    CUDAHOST_CALLABLE_MEMBER ~MonteRay_GridBins(void);

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRay_GridBins");}

    CUDAHOST_CALLABLE_MEMBER void init();

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void);

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRay_GridBins* rhs);

    void initialize( gpuRayFloat_t min, gpuRayFloat_t max, unsigned nBins);

    template<typename T>
    CUDAHOST_CALLABLE_MEMBER
    void
    initialize( const std::vector<T>& bins ) {
        verticesVec = new std::vector<gpuRayFloat_t>;
        verticesVec->resize( bins.size() );
        for( unsigned i = 0; i< bins.size(); ++i ) {
            verticesVec->at(i) = gpuRayFloat_t( bins[i] );
        }
        setup();
    }

    CUDAHOST_CALLABLE_MEMBER void setup(void);

    CUDA_CALLABLE_MEMBER unsigned getNumBins(void) const {
        //if( debug ) printf("Debug: MonteRay_GridBins::getNumBins -- \n");
        return numBins;
    }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getMinVertex(void) const {return minVertex; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getMaxVertex(void) const {return maxVertex; }
    CUDA_CALLABLE_MEMBER unsigned getNumVertices(void) const { return nVertices; }
    CUDA_CALLABLE_MEMBER unsigned getNumVerticesSq(void) const { return nVerticesSq; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getDelta(void) const { return delta; }

    CUDA_CALLABLE_MEMBER const gpuRayFloat_t* getVerticesData(void) const {return vertices;}

    void removeVertex(unsigned i);

    CUDA_CALLABLE_MEMBER void modifyForRadial(void);

    CUDA_CALLABLE_MEMBER bool isLinear(void) const { return type == LINEAR ? true : false; }
    CUDA_CALLABLE_MEMBER bool isRadial(void) const { return type == RADIAL ? true : false; }

    // returns -1 for one neg side of mesh
    // and number of bins on the pos side of the mesh
    CUDA_CALLABLE_MEMBER int getLinearIndex(gpuRayFloat_t pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuRayFloat_t r) const;
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuRayFloat_t rSq) const;

    CUDA_CALLABLE_MEMBER bool isIndexOutside( int i) const { return ( i < 0 ||  i >= getNumBins() ) ? true : false; }

    CUDA_CALLABLE_MEMBER inline
    gpuRayFloat_t distanceToGetInsideLinearMesh(const Position_t& pos, const Direction_t& dir, const int dim){
      gpuRayFloat_t dist = 0;
      if (pos[dim] >= this->vertices[this->getNumBins() - 1]){
        dist = dir[dim] < 0 ?
          (this->vertices[this->getNumBins() - 1] - pos[dim])/dir[dim] : 
          std::numeric_limits<gpuRayFloat_t>::infinity();
      } else if (pos[dim] <= this->vertices[0]) {
        dist = dir[dim] > 0 ?
          (this->vertices[0] - pos[dim])/dir[dim] : 
          std::numeric_limits<gpuRayFloat_t>::infinity();
      } else {
        return 0.0;
      }
      return dist + std::numeric_limits<gpuRayFloat_t>::epsilon(); 
    }

public:

    unsigned nVertices;
    unsigned nVerticesSq;

    std::vector<gpuRayFloat_t>* verticesVec = nullptr;
    std::vector<gpuRayFloat_t>* verticesSqVec = nullptr;

    gpuRayFloat_t* vertices;
    gpuRayFloat_t* verticesSq;

    gpuRayFloat_t delta;

private:
    gpuRayFloat_t minVertex;
    gpuRayFloat_t maxVertex;
    unsigned numBins;
    coordinate_t type;
    bool radialModified;

    CUDA_CALLABLE_MEMBER void validate();

    const bool debug = false;

public:
    void write( const std::string& fileName );
    void read( const std::string& fileName );

    void write(std::ostream& outfile) const;

    void read(std::istream& infile);
    void read_v0(std::istream& infile);

};

} /* namespace MonteRay */

#endif /* MonteRay_GridBins_HH_ */
