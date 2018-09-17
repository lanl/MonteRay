/*
 * MonteRay_SpatialGrid.hh
 *
 *  Created on: Feb 5, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_SPATIALGRID_HH_
#define MONTERAY_SPATIALGRID_HH_

#include "MonteRay_GridSystemInterface.hh"
#include "MonteRay_TransportMeshTypeEnum.hh"

//#include "Index.hh"
#include "MonteRayVector3D.hh"
//#include "Transformation.hh"

#include "MonteRayCopyMemory.hh"

namespace MonteRay {

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
    // rayTraceList_t -- defined in GridSystemInterface

    //TRA/JES Move to GridBins -- ?
    static const unsigned OUTSIDE_MESH;

    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid(void);
    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid( const MonteRay_SpatialGrid& );

    template<typename READER_T>
    CUDAHOST_CALLABLE_MEMBER MonteRay_SpatialGrid( READER_T& reader ) : MonteRay_SpatialGrid() {
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

    CUDAHOST_CALLABLE_MEMBER
    virtual ~MonteRay_SpatialGrid(void){

        if( ! Base::isCudaIntermediate ) {
            if( pGridInfo[0] ) delete pGridInfo[0];
            if( pGridInfo[1] ) delete pGridInfo[1];
            if( pGridInfo[2] ) delete pGridInfo[2];
            if( pGridSystem )  delete pGridSystem;
        }

    }

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRay_SpatialGrid");}

    CUDAHOST_CALLABLE_MEMBER void init() {
        CoordinateSystem = TransportMeshTypeEnum::NONE;
        dimension = 0;
        pGridInfo[0] = nullptr;
        pGridInfo[1] = nullptr;
        pGridInfo[2] = nullptr;
        initialized = false;
        pGridSystem = nullptr;
    }

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void) {
        //if( debug ) std::cout << "Debug: MonteRay_SpatialGrid::copyToGPU \n";
        if( ! initialized ) {
            throw std::runtime_error("MonteRay_SpatialGrid::copy -- MonteRay_SpatialGrid object has not been initialized.");
        }

        pGridInfo[0]->copyToGPU();
        pGridInfo[1]->copyToGPU();
        pGridInfo[2]->copyToGPU();

        pGridSystem->copyToGPU();

        Base::copyToGPU();
    }

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRay_SpatialGrid* rhs) {
        //		if( debug ) {
        //			std::cout << "Debug: MonteRay_SpatialGrid::copy(const MonteRay_SpatialGrid* rhs) \n";
        //		}

        if( ! rhs->initialized ) {
            throw std::runtime_error("MonteRay_SpatialGrid::copy -- MonteRay_SpatialGrid object has not been initialized.");
        }

        CoordinateSystem = rhs->CoordinateSystem;
        dimension = rhs->dimension;
        initialized = rhs->initialized;

#ifdef __CUDACC__
        if( isCudaIntermediate ) {
            // host to device
            pGridInfo[0] = rhs->pGridInfo[0]->devicePtr;
            pGridInfo[1] = rhs->pGridInfo[1]->devicePtr;
            pGridInfo[2] = rhs->pGridInfo[2]->devicePtr;
            pGridSystem = rhs->pGridSystem->getDeviceInstancePtr();

            if( debug ) {
                //				printf( "Debug: MonteRay_SpatialGrid::copy-- pGridInfo[%d]=%p \n", 0, pGridInfo[0] );
                //				printf( "Debug: MonteRay_SpatialGrid::copy-- pGridInfo[%d]=%p \n", 1, pGridInfo[1] );
                //				printf( "Debug: MonteRay_SpatialGrid::copy-- pGridInfo[%d]=%p \n", 2, pGridInfo[2] );
                //				printf( "Debug: MonteRay_SpatialGrid::copy--   pGridSystem=%p \n", pGridSystem );
            }

        } else {
            // device to host

        }


#else
        throw std::runtime_error("MonteRay_SpatialGrid::copy -- can NOT copy between host and device without CUDA.");
#endif
    }


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

    template<typename T>
    CUDAHOST_CALLABLE_MEMBER
    void
    setGrid( unsigned index, const std::vector<T>& vertices ) {
        checkDim( index+1 );
        if( pGridInfo[index] ) delete pGridInfo[index];
        pGridInfo[index] = new GridBins_t();
        pGridInfo[index]->initialize( vertices );
    }

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
    unsigned getIndex(Position_t particle_pos) const {
        MONTERAY_ASSERT_MSG( initialized, "MonteRay_SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            particle_pos = (*transform).counterTransformPos( particle_pos );
        //        }

        return  pGridSystem->getIndex( particle_pos );
    }

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t returnCellVolume( unsigned index ) const { return getVolume( index ); }

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getVolume( unsigned index ) const {
        MONTERAY_ASSERT( pGridSystem );
        return pGridSystem->getVolume( index );
    }

    template<class Particle>
    CUDA_CALLABLE_MEMBER
    void
    rayTrace(rayTraceList_t& rayTraceList, const Particle& p, gpuRayFloat_t distance, bool OutsideDistances=false) const {
        const bool debug = false;

        if( debug ) printf("MonteRay_SptialGrid::rayTrace(irayTraceList_t&, const Particle& p, gpuRayFloat distance, bool OutsideDistances\n");

        return rayTrace( rayTraceList, p.getPosition(), p.getDirection(), distance, OutsideDistances );
    }

    CUDA_CALLABLE_MEMBER
    void
    rayTrace(rayTraceList_t& rayTraceList, Position_t pos, Direction_t dir, gpuRayFloat_t distance, bool OutsideDistances=false) const {
        const bool debug = false;

        if( debug ) printf("MonteRay_SpatialGrid::rayTrace(rayTraceList_t&, Position_t pos, Direction_t dir, gpuRayFloat distance, bool OutsideDistances\n");

        MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            pos = (*transform).counterTransformPos( pos );
        //            dir = (*transform).counterTransformDir( dir );
        //        }
        if( debug ) printf("MonteRay_SpatialGrid::rayTrace -- calling grid system rayTrace \n");
        pGridSystem->rayTrace(rayTraceList, pos, dir, distance, OutsideDistances );
        return;
    }

    /// Call to support call with integer c array of indices
    /// and float c array of distances - may be slow
    CUDA_CALLABLE_MEMBER
    unsigned
    rayTrace(int* global_indices, gpuRayFloat_t* distances, Position_t pos, Direction_t dir, gpuRayFloat_t distance, bool OutsideDistances=false) const {
        const bool debug = false;

        if( debug ) printf("MonteRay_SpatialGrid::rayTrace(int* global_indices, int* gpuRayFloat_t* distances, Position_t pos, Direction_t dir, gpuRayFloat distance, bool OutsideDistances\n");
        MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            pos = (*transform).counterTransformPos( pos );
        //            dir = (*transform).counterTransformDir( dir );
        //        }
        rayTraceList_t rayTraceList;
        rayTrace(rayTraceList, pos, dir, distance, OutsideDistances );

        if( debug ) printf("MonteRay_SpatialGrid::rayTrace -- number of distances = %d\n", rayTraceList.size());
        for( unsigned i=0; i< rayTraceList.size(); ++i ) {
            global_indices[i] = rayTraceList.id(i);
            distances[i] = rayTraceList.dist(i);
        }
        return rayTraceList.size();
    }

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t& rayTraceMap, unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance) const {
        MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            pos = (*transform).counterTransformPos( pos );
        //            dir = (*transform).counterTransformDir( dir );
        //        }
        pGridSystem->crossingDistance(rayTraceMap, d, pos, dir, distance );
        return;
    }

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t& rayTraceMap, unsigned d, Position_t& pos, Direction_t& dir, gpuRayFloat_t distance) const {
        MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            pos = (*transform).counterTransformPos( pos );
        //            dir = (*transform).counterTransformDir( dir );
        //        }
        pGridSystem->crossingDistance(rayTraceMap, d, pos, dir, distance );
        return;
    }

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t& rayTraceMap, Position_t& pos, Direction_t& dir, gpuRayFloat_t distance) const {
        MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );

        //        if( transform ) {
        //            pos = (*transform).counterTransformPos( pos );
        //            dir = (*transform).counterTransformDir( dir );
        //        }
        pGridSystem->crossingDistance(rayTraceMap, pos, dir, distance );
        return;
    }

    //void setTransformation( const mcatk::Transformation& T);
    //const mcatk::Transformation* getTransformation( void ) const;
    //bool isTransformed( void ) const { return transform.is_initialized(); } //For testing purposes.

    //void transformPosDir( Position_t& Position, Direction_t& Direction) const;

    CUDA_CALLABLE_MEMBER
    std::string getGeometryType(void) const { return std::string("SpatialGrid"); }

    void write( const std::string& fileName );
    void read( const std::string& fileName );
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

} /* namespace MonteRay */

#endif /* MONTERAY_SPATIALGRID_HH_ */
