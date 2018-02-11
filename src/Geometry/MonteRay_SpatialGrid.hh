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

namespace MonteRay {

class MonteRay_SpatialGrid {
public:
    enum indexCartEnum_t {CART_X=0, CART_Y=1, CART_Z=2};
    enum indexCylEnum_t  {CYLR_R=0, CYLR_Z=1, CYLR_THETA=2};
    enum indexSphEnum_t  {SPH_R=0};
    enum { MaxDim=3 };

    using GridBins_t = MonteRay_GridBins;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    typedef Vector3D<gpuFloatType_t> Position_t;
    typedef Vector3D<gpuFloatType_t> Direction_t;
    // rayTraceList_t -- defined in GridSystemInterface

	//TRA/JES Move to GridBins -- ?
	static const unsigned OUTSIDE_MESH;

	MonteRay_SpatialGrid(void);
	MonteRay_SpatialGrid( const MonteRay_SpatialGrid& );
    virtual ~MonteRay_SpatialGrid(void){
    	if( pGridInfo[0] ) delete pGridInfo[0];
    	if( pGridInfo[1] ) delete pGridInfo[1];
    	if( pGridInfo[2] ) delete pGridInfo[2];
    }

    //  Disable assignment operator until needed
//    MonteRay_SpatialGrid& operator=( const MonteRay_SpatialGrid& );

    void initialize(void);

    TransportMeshTypeEnum::TransportMeshTypeEnum_t getCoordinateSystem(void) const {
        return CoordinateSystem;
    }
    void setCoordinateSystem(TransportMeshTypeEnum::TransportMeshTypeEnum_t system);

    unsigned getDimension(void) const { return dimension;}
    void setDimension( unsigned dim);

    void setGrid( unsigned index, gpuFloatType_t min, gpuFloatType_t max, unsigned numBins );
    void setGrid( unsigned index, const std::vector<gpuFloatType_t>& vertices );

    unsigned getNumGridBins( unsigned index ) const;
    gpuFloatType_t getMinVertex( unsigned index ) const;
    gpuFloatType_t getMaxVertex( unsigned index ) const;
    gpuFloatType_t getDelta(unsigned index) const;

    gpuFloatType_t getVertex(unsigned d, unsigned i ) const;
    //const std::vector<gpuFloatType_t>& getVertices( unsigned d) const { return gridInfo.at(d).vertices; }

    unsigned getNumCells(void) const;
    size_t numCells(void) const {return getNumCells();}
    size_t size(void) const {return getNumCells();}

    bool isInitialized(void) const { return initialized; }

    //typedef mcatk::Index<1> Index_t;
    typedef unsigned Index_t;
    template <typename Particle_t>
    Index_t operator()( const Particle_t& p ) { return getIndex(p); }
    size_t footprint( void ) const { return 0 /*sizeof( what? )*/; }

    template<class Particle>
    unsigned getIndex(const Particle& p) const {
        Position_t particle_pos = p.getPosition();
        return getIndex( particle_pos );
    }

    unsigned getIndex(Position_t particle_pos) const {
    	MONTERAY_ASSERT_MSG( initialized, "MonteRay_SpatialGrid MUST be initialized before tying to get an index." );

//        if( transform ) {
//            particle_pos = (*transform).counterTransformPos( particle_pos );
//        }

        return  pGridSystem->getIndex( particle_pos );
    }

    gpuFloatType_t returnCellVolume( unsigned index ) const { return getVolume( index ); }

    gpuFloatType_t getVolume( unsigned index ) const {
        MONTERAY_ASSERT( pGridSystem );
        return pGridSystem->getVolume( index );
    }

    template<class Particle>
    rayTraceList_t
    rayTrace( const Particle& p, gpuFloatType_t distance, bool OutsideDistances=false) const {
        return rayTrace( p.getPosition(), p.getDirection(), distance, OutsideDistances );
    }

    rayTraceList_t
    rayTrace( Position_t pos, Direction_t dir, gpuFloatType_t distance, bool OutsideDistances=false) const {
    	if( debug ) printf( "Debug: MonteRay_SpatialGrid::rayTrace -- \n");
    	MONTERAY_ASSERT_MSG( initialized, "SpatialGrid MUST be initialized before tying to get an index." );
    	rayTraceList_t rayTraceList;

//        if( transform ) {
//            pos = (*transform).counterTransformPos( pos );
//            dir = (*transform).counterTransformDir( dir );
//        }
        pGridSystem->rayTrace(rayTraceList, pos, dir, distance, OutsideDistances );
        return rayTraceList;

    }


    //void setTransformation( const mcatk::Transformation& T);
    //const mcatk::Transformation* getTransformation( void ) const;
    //bool isTransformed( void ) const { return transform.is_initialized(); } //For testing purposes.

    //void transformPosDir( Position_t& Position, Direction_t& Direction) const;

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
    inline void checkDim( unsigned dim ) const;

public:
    void write(std::ostream& outfile) const;

    void  read(std::istream& infile);
    void  read_v0(std::istream& infile);

};

} /* namespace MonteRay */

#endif /* MONTERAY_SPATIALGRID_HH_ */
