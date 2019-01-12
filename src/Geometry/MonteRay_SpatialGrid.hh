#ifndef MONTERAY_SPATIALGRID_HH_
#define MONTERAY_SPATIALGRID_HH_

#include "MonteRay_TransportMeshTypeEnum.hh"

//#include "Index.hh"
#include "MonteRayVector3D.hh"
//#include "Transformation.hh"

#include "MonteRayCopyMemory.hh"

namespace MonteRay {

class MonteRay_GridSystemInterface;
class MonteRay_GridBins;
class RayWorkInfo;

class MonteRay_SpatialGrid : public CopyMemoryBase<MonteRay_SpatialGrid> {
public:
    using Base = MonteRay::CopyMemoryBase<MonteRay_SpatialGrid> ;

    enum indexCartEnum_t {CART_X=0, CART_Y=1, CART_Z=2};
    enum indexCylEnum_t  {CYLR_R=0, CYLR_Z=1, CYLR_THETA=2};
    enum indexSphEnum_t  {SPH_R=0};
    enum { MaxDim=3 };

    using GridBins_t = MonteRay_GridBins;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    typedef Vector3D<gpuRayFloat_t> Position_t;
    typedef Vector3D<gpuRayFloat_t> Direction_t;

    //TRA/JES Move to GridBins -- ?
    static const unsigned OUTSIDE_MESH;

    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid(void);
    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid( const MonteRay_SpatialGrid& );

    template<typename READER_T>
    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid( READER_T& reader );

    CUDAHOST_CALLABLE_MEMBER
    ~MonteRay_SpatialGrid(void);

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRay_SpatialGrid");}

    CUDAHOST_CALLABLE_MEMBER void init();

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void);

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRay_SpatialGrid* rhs);


    //  Disable assignment operator until needed
    //    MonteRay_SpatialGrid& operator=( const MonteRay_SpatialGrid& );

    CUDAHOST_CALLABLE_MEMBER
    void initialize(void);

    CUDA_CALLABLE_MEMBER
    TransportMeshTypeEnum::TransportMeshTypeEnum_t getCoordinateSystem(void) const {
        return CoordinateSystem;
    }

    CUDAHOST_CALLABLE_MEMBER
    void setCoordinateSystem(TransportMeshTypeEnum::TransportMeshTypeEnum_t system);

    CUDA_CALLABLE_MEMBER
    unsigned getDimension(void) const { return dimension;}

    CUDAHOST_CALLABLE_MEMBER
    void setDimension( unsigned dim);

    CUDAHOST_CALLABLE_MEMBER
    void setGrid( unsigned index, gpuRayFloat_t min, gpuRayFloat_t max, unsigned numBins );

    CUDAHOST_CALLABLE_MEMBER
    void setGrid( unsigned index, const std::vector<double>& vertices );

    CUDAHOST_CALLABLE_MEMBER
    void setGrid( unsigned index, const std::vector<float>& vertices );

    CUDA_CALLABLE_MEMBER
    unsigned getNumGridBins( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getMinVertex( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getMaxVertex( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getDelta(unsigned index) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getVertex(unsigned d, unsigned i ) const;
    //const std::vector<gpuRayFloat_t>& getVertices( unsigned d) const { return gridInfo.at(d).vertices; }

    CUDA_CALLABLE_MEMBER
    unsigned getNumCells(void) const;

    CUDA_CALLABLE_MEMBER
    size_t numCells(void) const {return getNumCells();}

    CUDA_CALLABLE_MEMBER
    size_t getNumVertices(unsigned i) const;

    CUDA_CALLABLE_MEMBER
    size_t getNumVerticesSq(unsigned i) const;

    CUDA_CALLABLE_MEMBER
    size_t size(void) const {return getNumCells();}

    CUDA_CALLABLE_MEMBER
    bool isInitialized(void) const { return initialized; }

    //typedef mcatk::Index<1> Index_t;
    typedef unsigned Index_t;

    template <typename Particle_t>
    CUDA_CALLABLE_MEMBER
    Index_t operator()( const Particle_t& p ) { return getIndex(p); }

    CUDA_CALLABLE_MEMBER
    size_t footprint( void ) const { return 0 /*sizeof( what? )*/; }

    template<class Particle>
    CUDA_CALLABLE_MEMBER
    unsigned getIndex(const Particle& p) const {
        Position_t particle_pos = p.getPosition();
        return getIndex( particle_pos );
    }

    CUDA_CALLABLE_MEMBER
    unsigned getIndex(Position_t particle_pos) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t returnCellVolume( unsigned index ) const { return getVolume( index ); }

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getVolume( unsigned index ) const;

    template<class Particle>
    CUDA_CALLABLE_MEMBER
    void
    rayTrace(  const unsigned threadID,
               RayWorkInfo& rayInfo,
               const Particle& p,
               const gpuRayFloat_t distance,
               const bool OutsideDistances=false ) const {

        const bool debug = false;

        if( debug ) printf("MonteRay_SptialGrid::rayTrace(threadID, RayWorkInfo&, Particle& , float_t distance, bool OutsideDistances\n");

        rayTrace( threadID, rayInfo, p.getPosition(), p.getDirection(), distance, OutsideDistances );
    }

    CUDA_CALLABLE_MEMBER
    unsigned
    rayTrace( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const Position_t& pos,
              const Position_t& dir,
              const gpuRayFloat_t distance,
              const bool outsideDistances=false) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance( const unsigned dim,
                      const unsigned threadID,
                      RayWorkInfo& rayInfo,
                      const gpuRayFloat_t pos,
                      const gpuRayFloat_t dir,
                      const gpuRayFloat_t distance) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance( const unsigned dim,
                      const unsigned threadID,
                      RayWorkInfo& rayInfo,
                      const Position_t& pos,
                      const Direction_t& dir,
                      const gpuRayFloat_t distance) const;

    //void setTransformation( const mcatk::Transformation& T);
    //const mcatk::Transformation* getTransformation( void ) const;
    //bool isTransformed( void ) const { return transform.is_initialized(); } //For testing purposes.

    //void transformPosDir( Position_t& Position, Direction_t& Direction) const;

    CUDA_CALLABLE_MEMBER
    std::string getGeometryType(void) const { return std::string("SpatialGrid"); }

    void write( const std::string& fileName ) const;
    void read( const std::string& fileName );

    void writeToFile( const std::string& fileName ) const { write(fileName); }
    void readFromFile( const std::string& fileName ) { read(fileName); }

private:
    TransportMeshTypeEnum::TransportMeshTypeEnum_t CoordinateSystem = TransportMeshTypeEnum::NONE;
    unsigned dimension = 0;
    pArrayOfpGridInfo_t pGridInfo;

    bool initialized = false;

    MonteRay_GridSystemInterface* pGridSystem = nullptr;

    //boost::optional<mcatk::Transformation> transform; // optional transformation of the grid

    const bool debug = false;

private:
    CUDA_CALLABLE_MEMBER
    void checkDim( unsigned dim ) const;

public:
    void write(std::ostream& outfile) const;

    void  read(std::istream& infile);
    void  read_v0(std::istream& infile);

};

template<typename READER_T>
CUDAHOST_CALLABLE_MEMBER
MonteRay_SpatialGrid::MonteRay_SpatialGrid( READER_T& reader ) : MonteRay_SpatialGrid() {
    if( reader.getGeometryString() == "XYZ" )  {
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
        const unsigned DIM=3;
        setDimension(DIM);
        for( unsigned d=0; d < DIM; ++d) {
            std::vector<double> vertices = reader.getVertices(d);
            setGrid(d, vertices );
        }
        initialize();
    }else if( reader.getGeometryString() == "RZ" )  {
        setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );
        const unsigned DIM=2;
        setDimension(DIM);
        for( unsigned d=0; d < DIM; ++d) {
            std::vector<double> vertices = reader.getVertices(d);
            setGrid(d, vertices );
        }
        initialize();
    } else {
        throw std::runtime_error( "MonteRay_SpatialGrid(reader) -- Geometry type not yet supported." );
    }
}

} /* namespace MonteRay */

#endif /* MONTERAY_SPATIALGRID_HH_ */
