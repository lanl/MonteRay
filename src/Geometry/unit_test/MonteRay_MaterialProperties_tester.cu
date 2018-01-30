#include <UnitTest++.h>

#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_SetupMaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"

#include <iostream>
#include <vector>
#include <cmath>

#include <cassert>

namespace MaterialProperties_tester{
using namespace std;
using namespace MonteRay;

typedef MonteRay_CellProperties CellProperties;
typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;
typedef MonteRay_MaterialSpec MaterialSpec;
typedef MonteRay_MaterialProperties MaterialProperties;

template<class T>
using SetupMaterialProperties = MonteRay_SetupMaterialProperties<T>;

SUITE( MaterialProperties_tests ) {

    typedef std::vector< MonteRay_MaterialSpec > MatList_t;
    double epsilon = 1e-13;

    class MaterialProperties_tester : public MonteRay_MaterialProperties {
    public:
        MaterialProperties_tester() : MaterialProperties() {};
        ~MaterialProperties_tester(){};

        CellProperties getCellTester( Cell_Index_t cellID ) const {
            return getCell(cellID);
        }

        template<typename SomeParticle_t>
        MaterialProperties::Cell_Index_t getCellIndexTester(const SomeParticle_t& p) const {
            return getCellIndex(p);
        }
    };

    TEST(Ctor_withLnk3dnt){
        ReadLnk3dnt readerObject( "lnk3dnt/3iso_3shell_godiva.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties problemMaterialProperties;
        problemMaterialProperties.setMaterialDescription( readerObject );

        CHECK_EQUAL(size_t(8000), problemMaterialProperties.size());
        CHECK_EQUAL(size_t(8000), problemMaterialProperties.getNTotalCells());

        int cellNumber = 7010;

        CHECK_EQUAL( size_t(2) , problemMaterialProperties.getNumMaterials(cellNumber) );
        CHECK_EQUAL( 2         , problemMaterialProperties.getMaterialID(cellNumber,0) );
        CHECK_EQUAL( 3         , problemMaterialProperties.getMaterialID(cellNumber,1) );
        CHECK_CLOSE( 17.8772   , problemMaterialProperties.getMaterialDensity(cellNumber,0), 1e-04);
        CHECK_CLOSE(  0.8228   , problemMaterialProperties.getMaterialDensity(cellNumber,1), 1e-04);

        cellNumber = 7410;
        CHECK_EQUAL( size_t(2) , problemMaterialProperties.getNumMaterials(cellNumber) );
        CHECK_EQUAL( 3         , problemMaterialProperties.getMaterialID(cellNumber,0) );
        CHECK_EQUAL( 4         , problemMaterialProperties.getMaterialID(cellNumber,1) );
        CHECK_CLOSE( 8.6768    , problemMaterialProperties.getMaterialDensity(cellNumber,0), 1e-04);
        CHECK_CLOSE( 4.4132    , problemMaterialProperties.getMaterialDensity(cellNumber,1), 1e-04);

    }

    TEST(Ctor_IDsDensities){

        vector<int> IDs;
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(2);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(2.0);
        densities.push_back(2.0);

        size_t TotalNumberOfCells = 5;
        MaterialProperties MP1( IDs, densities, TotalNumberOfCells );

        int cellNumber = 3;
        CHECK_EQUAL( size_t(1) , MP1.getNumMaterials(cellNumber) );
        CHECK_EQUAL( 2         , MP1.getMaterialID(cellNumber,0) );
        CHECK_CLOSE( 2.0       , MP1.getMaterialDensity(cellNumber,0), 1e-04);
    }

    TEST( clear ){
        vector<int> IDs;
        IDs.push_back(1);
        IDs.push_back(1);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);

        size_t TotalNumberOfCells = 2;
        MaterialProperties mp( IDs, densities, TotalNumberOfCells );

        CHECK_EQUAL( 2, mp.size() );
        mp.clear();
        CHECK_EQUAL( 0, mp.size() );
    }

    TEST(add_cell) {
        CellProperties cell;
        cell.add( 1, 0.1);
        cell.add( 2, 0.2);

        MaterialProperties mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell );
        CHECK_EQUAL( 1, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);
    }

    class TestParticle_t {
    public:
        TestParticle_t(){
            cellIndex = -2;
        };
        ~TestParticle_t(){};

        int cellIndex;
        void setLocationIndex( int index) { cellIndex = index; }
        MaterialProperties::Cell_Index_t getLocationIndex(void) const { return cellIndex; }
    };

    TEST(getCellIndex) {
        MaterialProperties_tester mp;
        CellProperties cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);
        mp.add(cell1);

        TestParticle_t p;
        p.setLocationIndex( 0 );
        CHECK_EQUAL( 0, mp.getCellIndexTester<TestParticle_t>(p) );
    }

    TEST(getCellIndex_fails_no_cells) {
        MaterialProperties_tester mp;

         TestParticle_t p;
         p.setLocationIndex( 0 );
#ifdef BOOST_ENABLE_ASSERT_HANDLER
         CHECK_THROW( mp.getCellIndexTester<TestParticle_t>(p), std::runtime_error );
#endif
     }

    TEST(getCellIndex_fails_index_neg2) {
        // -2 is the default value, particle is not get assigned a location index
        MaterialProperties_tester mp;

        TestParticle_t p;
        p.setLocationIndex( -2 );
#ifdef BOOST_ENABLE_ASSERT_HANDLER
        CHECK_THROW( mp.getCellIndexTester<TestParticle_t>(p), std::exception );
#endif
    }

    TEST(getCellIndex_neg1_returns_0) {
        // -1 indicates a solid body geometry node
        MaterialProperties_tester mp;
        CellProperties cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);
        mp.add(cell1);

        TestParticle_t p;
        p.setLocationIndex( -1 );
        CHECK_EQUAL( 0, mp.getCellIndexTester<TestParticle_t>(p) );
    }

    TEST(getNumMaterial_from_Particle_and_index){
        CellProperties cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);

        CellProperties cell2;
        cell2.add( 1, 0.1);
        cell2.add( 2, 0.2);
        cell2.add( 3, 0.3);

        MaterialProperties mp;
        mp.add(cell1);
        mp.add(cell2);

        TestParticle_t p;
        p.setLocationIndex( 0 );

        CHECK_EQUAL( 2, mp.getNumMaterials(0) );
        CHECK_EQUAL( 2, mp.getNumMaterials<TestParticle_t>(p) );
     }

    TEST(getMaterialDensity_and_MatID_and_temp_from_Particle_and_index){
        CellProperties cell1;
        cell1.add( 10, 10.1);
        cell1.add( 20, 20.2);
        cell1.setTemperature( 99.0 );

        CellProperties cell2;
        cell2.add( 100, 100.1);
        cell2.add( 200, 200.2);
        cell2.add( 300, 300.3);
        cell2.setTemperature( 199.0 );

        MaterialProperties mp;
        mp.add(cell1);
        mp.add(cell2);

        TestParticle_t p;
        p.setLocationIndex( 0 );

        int cellIndex = 0;
        CHECK_EQUAL( 10, mp.getMaterialID<TestParticle_t>(p,0) );
        CHECK_EQUAL( 10, mp.getMaterialID(cellIndex,0) );
        CHECK_CLOSE( 10.1, mp.getMaterialDensity<TestParticle_t>(p,0), 1e-6 );
        CHECK_CLOSE( 10.1, mp.getMaterialDensity(cellIndex,0), 1e-6 );
        CHECK_CLOSE( 99.0, mp.getTemperature<TestParticle_t>(p), 1e-6 );
        CHECK_CLOSE( 99.0, mp.getTemperature(cellIndex), 1e-11 );

        p.setLocationIndex(1);
        cellIndex = 1;
        CHECK_EQUAL( 100, mp.getMaterialID<TestParticle_t>(p,0) );
        CHECK_EQUAL( 100, mp.getMaterialID(cellIndex,0) );
        CHECK_CLOSE( 100.1, mp.getMaterialDensity<TestParticle_t>(p,0), 1e-5 );
        CHECK_CLOSE( 100.1, mp.getMaterialDensity(cellIndex,0), 1e-5 );
        CHECK_CLOSE( 199.0, mp.getTemperature<TestParticle_t>(p), 1e-11 );
        CHECK_CLOSE( 199.0, mp.getTemperature(cellIndex), 1e-11 );
     }

    TEST(SetGlobalTemperature){

        vector<int> IDs;
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(2);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(2.0);
        densities.push_back(2.0);

        size_t TotalNumberOfCells = 5;
        MaterialProperties MP1( IDs, densities, TotalNumberOfCells );
        MP1.setGlobalTemperature( 0.001 );

        CHECK_CLOSE(0.001 , MP1.getTemperature(3), epsilon );
    }
    TEST(SetIndividualTemperature){

         vector<int> IDs;
         IDs.push_back(1);
         IDs.push_back(1);
         IDs.push_back(1);
         IDs.push_back(2);
         IDs.push_back(2);

         vector<double> densities;
         densities.push_back(1.0);
         densities.push_back(1.0);
         densities.push_back(1.0);
         densities.push_back(2.0);
         densities.push_back(2.0);

         size_t TotalNumberOfCells = 5;
         MaterialProperties MP1( IDs, densities, TotalNumberOfCells );
         MP1.setCellTemperature( 0, 0.001 );
         MP1.setCellTemperature( 1, 0.002 );
         MP1.setCellTemperature( 2, 0.003 );
         MP1.setCellTemperature( 3, 0.004 );
         MP1.setCellTemperature( 4, 0.005 );

         CHECK_CLOSE(0.001 , MP1.getTemperature(0), epsilon );
         CHECK_CLOSE(0.002 , MP1.getTemperature(1), epsilon );
         CHECK_CLOSE(0.003 , MP1.getTemperature(2), epsilon );
         CHECK_CLOSE(0.004 , MP1.getTemperature(3), epsilon );
         CHECK_CLOSE(0.005 , MP1.getTemperature(4), epsilon );
         
         // Disabled in MonteRay
//         MP1.setCellTemperatureCelsius( 0, 30.0 );
//         CHECK_CLOSE(30.0 , MP1.getTemperatureCelsius(0), epsilon );
     }

    TEST(SetCellTemperatures){

        vector<int> IDs;
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(2);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(2.0);
        densities.push_back(2.0);

        size_t TotalNumberOfCells = 5;
        MaterialProperties MP1( IDs, densities, TotalNumberOfCells );

        vector<double> temperatures;
        temperatures.push_back(0.001);
        temperatures.push_back(0.001);
        temperatures.push_back(0.001);
        temperatures.push_back(0.002);

        // Fail on number of temps not matching the number of cells
#ifdef BOOST_ENABLE_ASSERT_HANDLE
        CHECK_THROW( MP1.setCellTemperatures( temperatures ), std::runtime_error );
#endif

        // Should be safe after one more temperature
        temperatures.push_back(0.002);
        MP1.setCellTemperatures( temperatures );

        CHECK_CLOSE(0.001, MP1.getTemperature(0), epsilon );
        CHECK_CLOSE(0.001, MP1.getTemperature(1), epsilon );
        CHECK_CLOSE(0.001, MP1.getTemperature(2), epsilon );
        CHECK_CLOSE(0.002, MP1.getTemperature(3), epsilon );
        CHECK_CLOSE(0.002, MP1.getTemperature(4), epsilon );
#ifdef BOOST_ENABLE_ASSERT_HANDLE
        CHECK_CLOSE(0.002, MP1.getTemperature(5), epsilon );
#endif
    }

    TEST(ScaleDensities){

        vector<int> IDs;
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(2);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(1.0);
        densities.push_back(2.0);
        densities.push_back(2.0);

        size_t TotalNumberOfCells = 5;
        MaterialProperties MP1( IDs, densities, TotalNumberOfCells );

        CHECK_EQUAL( size_t(1)  , MP1.getNumMaterials(3) );
        CHECK_EQUAL( 2          , MP1.getMaterialID(3,0) );
        CHECK_EQUAL( 2.0        , MP1.getMaterialDensity(3,0) );

        MP1.scaleMaterialDensity(2,9.0);
        CHECK_EQUAL(18.0        , MP1.getMaterialDensity(3,0) );

        MP1.scaleAllMaterialDensities(0.1);

        CHECK_CLOSE( 0.1       , MP1.getMaterialDensity(0,0), 1e-04);
        CHECK_CLOSE( 0.1       , MP1.getMaterialDensity(1,0), 1e-04);
        CHECK_CLOSE( 0.1       , MP1.getMaterialDensity(2,0), 1e-04);
        CHECK_CLOSE( 1.8       , MP1.getMaterialDensity(3,0), 1e-04);
        CHECK_CLOSE( 1.8       , MP1.getMaterialDensity(4,0), 1e-04);
    }
    TEST(setSingleMaterial){

        const int ID = 99;
        const double density = 25.0;
        MaterialSpec MS(ID, density);

        const int NumberOfCells = 20;
        SetupMaterialProperties<double> MP;

        MP.setSingleMaterial(MS,NumberOfCells);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        int cell = 5;
        CHECK_EQUAL( ID          , MP.getMaterialID(cell,0) );
        CHECK_EQUAL( density     , MP.getMaterialDensity(cell,0) );
    }


    TEST(AddCellMaterial){

        const int ID = 99;
        const double density = 25.0;
        MaterialSpec MS(ID, density);

        const int NumberOfCells = 20;
        SetupMaterialProperties<double> MP;

        MP.setSingleMaterial(MS,NumberOfCells);
        const unsigned cellID = 5;
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MaterialSpec MSnew(IDnew, densityNew);

        MP.addCellMaterial(MSnew,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );
    }
    TEST( RemoveCellMaterial ){

        const int ID = 99;
        const double density = 25.0;
        MaterialSpec MS(ID, density);

        const int NumberOfCells = 20;
        SetupMaterialProperties<double> MP;

        MP.setSingleMaterial(MS,NumberOfCells);
        const unsigned cellID = 5;
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MaterialSpec MSnew(IDnew, densityNew);

        MP.addCellMaterial(MSnew,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( size_t(2), MP.getNumMaterials(cellID) );

        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        MP.removeCellMaterial(MS,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        CHECK_EQUAL( size_t(1) , MP.getNumMaterials(cellID) );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,0), 1e-05 );

   }
    TEST( ReplaceCellMaterial ){

        const int ID = 99;
        const double density = 25.0;
        MaterialSpec MS(ID, density);

        const int NumberOfCells = 20;
        SetupMaterialProperties<double> MP;

        MP.setSingleMaterial(MS,NumberOfCells);
        const unsigned cellID = 5;
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MaterialSpec MSnew(IDnew, densityNew);

        MP.addCellMaterial(MSnew,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        CHECK_EQUAL( size_t(2), MP.getNumMaterials(cellID) );
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        const int    IDreplace      = 45;
        const double densityReplace = 55.0;
        MaterialSpec MSreplace(IDreplace, densityReplace);

        MP.replaceCellMaterial(MS,MSreplace,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( size_t(2),       MP.getNumMaterials(cellID) );
        CHECK_EQUAL( IDnew          , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densityNew     , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDreplace      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityReplace , MP.getMaterialDensity(cellID,1), 1e-05 );
   }

    class reNumberTester{
    public:
    	reNumberTester(){}
    	~reNumberTester(){}

    	std::vector<int> idList;


        unsigned materialIDtoIndex(unsigned id) const {
            for( unsigned i=0; i < idList.size(); ++i ){
                if( id == idList[i] ) {
                    return i;
                }
            }

            throw std::runtime_error( "reNumberTester::materialIDtoIndex -- id not found.");
        }

    };

    TEST( ReNumberAllIDs_by_function ){
        const int NumberOfCells = 20;
        std::vector<MaterialProperties::MatID_t> IDs( NumberOfCells, 99);
        IDs[0] = 50;

        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties MP( IDs, densities, NumberOfCells );

        CHECK_EQUAL( 50, MP.getMaterialID(0,0) );
        CHECK_EQUAL( 99, MP.getMaterialID(1,0) );
        CHECK_EQUAL( 99, MP.getMaterialID(19,0) );

        reNumberTester renumberer;
        renumberer.idList.push_back( 50 );
        renumberer.idList.push_back( 99 );

        MP.renumberMaterialIDs( renumberer );

        CHECK_EQUAL( 0, MP.getMaterialID(0,0) );
        CHECK_EQUAL( 1, MP.getMaterialID(1,0) );
        CHECK_EQUAL( 1, MP.getMaterialID(19,0) );
    }

    TEST( ContainsMaterial ){
        const int ID = 99;
        const double density = 25.0;
        const int IDnew = 9;
        const double densityNew = 75.0;

        MaterialSpec MS(ID, density);

        const int NumberOfCells = 20;
        SetupMaterialProperties<double> MP;

        MP.setSingleMaterial(MS,NumberOfCells);
        const unsigned cellID = 5;
        CHECK_EQUAL( ID          , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density     , MP.getMaterialDensity(cellID,0), 1e-05 );

        CHECK( MP.containsMaterial(cellID,ID) ); // contains the first material
        CHECK( !MP.containsMaterial(cellID,IDnew) ); // doesn't contain the second material yet

        MaterialSpec MSnew(IDnew, densityNew);

        MP.addCellMaterial(MSnew,cellID);
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        CHECK_EQUAL( size_t(2),   MP.getNumMaterials(cellID) );
        CHECK_EQUAL( ID         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( density    , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        CHECK( MP.containsMaterial(cellID,ID) ); // contains the first material
        CHECK( MP.containsMaterial(cellID,IDnew) ); // contains the second material
    }

    TEST( scale_CellProperties_Density ){
        CellProperties cell;

        cell.add( -1, 0.1 );
        cell.add(  2, 0.2 );
        cell.add(  3, 0.3 );

        CHECK_EQUAL( 3, cell.size() );
        CHECK_CLOSE( 0.1, cell.getMaterialDensity(0), 1e-5 );
        cell.scaleDensity(-1, 10.0 );
        CHECK_CLOSE( 1.0, cell.getMaterialDensity(0), 1e-11 );
    }

    double XSec( int id, double den, double temp ) {
        if( id == 1 ) {
            return 1.0*den;
        }
        if( id == 2 ) {
            return 10.0*den;
        }
        if( id == 3 ) {
            return 100.0*den;
        }
        return 0.0;
    }

    class testParticle {
    public:
        testParticle(){};
        ~testParticle(){};

        int getLocationIndex(void) const { return 0; }
    };

    TEST( getXsecSum ){
        CellProperties cell;

        auto XSec_Func = [&] ( int ID, double den, double temp ) {
            return XSec( ID, den, temp );
        };

        cell.add(  1, 0.1 );
        cell.add(  2, 0.2 );
        cell.add(  3, 0.3 );

        MaterialProperties mp;
        mp.add( cell );
        testParticle p;

        double expected = 1.0*0.1 + 10.0*0.2 + 100.0*0.3;

        double xsec = mp.getXsecSum( XSec_Func, p );
        CHECK_CLOSE( expected, xsec, 1e-5);
    }


    TEST( getCellData_from_getCell_assignment ){
        CellProperties cell;

        cell.add(  1, 0.1 );
        cell.add(  2, 0.2 );
        cell.add(  3, 0.3 );
        cell.setTemperature( 100.1 );

        MaterialProperties_tester mp;
        mp.add( cell );

        CHECK_EQUAL( 3, mp.getNumMaterials(0) );
        CHECK_CLOSE( 0.3, mp.getMaterialDensity(0,2), 1e-5);
        CHECK_CLOSE( 100.1, mp.getTemperature(0), 1e-11);

        CellProperties cell2 = mp.getCellTester(0);
        CHECK_EQUAL( 3, cell2.getNumMaterials() );
        CHECK_CLOSE( 0.3, cell2.getMaterialDensity(2), 1e-5);
        CHECK_CLOSE( 100.1, cell2.getTemperature(), 1e-11);
    }

    TEST( getCellData_from_getCell_copyCtor ){
        CellProperties cell;

        cell.add(  1, 0.1 );
        cell.add(  2, 0.2 );
        cell.add(  3, 0.3 );
        cell.setTemperature( 100.1 );

        MaterialProperties_tester mp;
        mp.add( cell );

        CHECK_EQUAL( 3, mp.getNumMaterials(0) );
        CHECK_CLOSE( 0.3, mp.getMaterialDensity(0,2), 1e-5);
        CHECK_CLOSE( 100.1, mp.getTemperature(0), 1e-11);

        CellProperties cell2( mp.getCellTester(0) );
        CHECK_EQUAL( 3, cell2.getNumMaterials() );
        CHECK_CLOSE( 0.3, cell2.getMaterialDensity(2), 1e-5);
        CHECK_CLOSE( 100.1, cell2.getTemperature(), 1e-11);
    }

    TEST( addCellMaterial ) {
        CellProperties cell;

        cell.add(  1, 0.1 );
        cell.add(  2, 0.2 );
        cell.setTemperature( 100.1 );

        MaterialProperties_tester mp;
        mp.add( cell );
        CHECK_EQUAL( 2, mp.getNumMaterials(0) );

        unsigned cellID = 0;
        mp.addCellMaterial( 0, 3, 0.3);

        CHECK_EQUAL( 3, mp.getNumMaterials(0) );
        CHECK_CLOSE( 0.3, mp.getMaterialDensity(0,2), 1e-5);
    }

    TEST( addCellMaterial_fails_no_such_cell ) {
        CellProperties cell;

        cell.add(  1, 0.1 );
        cell.add(  2, 0.2 );
        cell.setTemperature( 100.1 );

        MaterialProperties_tester mp;
        mp.add( cell );
        CHECK_EQUAL( 2, mp.getNumMaterials(0) );

        unsigned cellID = 0;
#ifdef BOOST_ENABLE_ASSERT_HANDLE
        CHECK_THROW( mp.addCellMaterial( 1, 3, 0.3), std::runtime_error );
#endif
    }

    TEST( addEmptyCells ) {
        MaterialProperties_tester mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add();
        CHECK_EQUAL( 1, mp.size() );
        mp.add();
        CHECK_EQUAL( 2, mp.size() );
    }

    TEST( emptyCell_size ) {
        MaterialProperties_tester mp;
        mp.add();
        CHECK_EQUAL( 0, mp.getNumMaterials(0) );
    }

    TEST( add_materials_to_empty_cell ) {
        MaterialProperties_tester mp;
        mp.add();

        mp.addCellMaterial( 0, 10, 10.1);
        CHECK_EQUAL( 1, mp.getNumMaterials(0) );
        CHECK_CLOSE( 10, mp.getMaterialID(0,0), 1e-11);
        CHECK_CLOSE( 10.1, mp.getMaterialDensity(0,0), 1e-5);
    }

    TEST(  initializeMaterialDescription ) {
        std::vector<int> matid;
        std::vector<double> density;

        unsigned NTotalCells = 3;
        unsigned NumMatsPerCell = 5;
        for( unsigned i = 0; i < NumMatsPerCell*NTotalCells; ++i ) {
            matid.push_back( i );
            density.push_back( double(i) + 0.1 );
        }

        MaterialProperties mp;
        mp.initializeMaterialDescription( matid, density, NTotalCells );

        CHECK_EQUAL( 3, mp.size() );
        CHECK_EQUAL( 5, mp.getNumMaterials(0) );

        int cell = 0;
        CHECK_EQUAL(  0, mp.getMaterialID(cell,0) );
        CHECK_CLOSE(  0.1, mp.getMaterialDensity(cell,0), 1e-5 );
        CHECK_EQUAL(  3, mp.getMaterialID(cell,1) );
        CHECK_CLOSE(  3.1, mp.getMaterialDensity(cell,1), 1e-5 );
        CHECK_EQUAL(  6, mp.getMaterialID(cell,2) );
        CHECK_CLOSE(  6.1, mp.getMaterialDensity(cell,2), 1e-5 );
        CHECK_EQUAL(  9, mp.getMaterialID(cell,3) );
        CHECK_CLOSE(  9.1, mp.getMaterialDensity(cell,3), 1e-5 );
        CHECK_EQUAL(  12, mp.getMaterialID(cell,4) );
        CHECK_CLOSE(  12.1, mp.getMaterialDensity(cell,4), 1e-5 );

        cell = 1;
        CHECK_EQUAL(  1, mp.getMaterialID(cell,0) );
        CHECK_CLOSE(  1.1, mp.getMaterialDensity(cell,0), 1e-5 );
        CHECK_EQUAL(  4, mp.getMaterialID(cell,1) );
        CHECK_CLOSE(  4.1, mp.getMaterialDensity(cell,1), 1e-5 );
        CHECK_EQUAL(  7, mp.getMaterialID(cell,2) );
        CHECK_CLOSE(  7.1, mp.getMaterialDensity(cell,2), 1e-5 );
        CHECK_EQUAL(  10, mp.getMaterialID(cell,3) );
        CHECK_CLOSE(  10.1, mp.getMaterialDensity(cell,3), 1e-5 );
        CHECK_EQUAL(  13, mp.getMaterialID(cell,4) );
        CHECK_CLOSE(  13.1, mp.getMaterialDensity(cell,4), 1e-5 );

        cell = 2;
        CHECK_EQUAL(  2, mp.getMaterialID(cell,0) );
        CHECK_CLOSE(  2.1, mp.getMaterialDensity(cell,0), 1e-5 );
        CHECK_EQUAL(  5, mp.getMaterialID(cell,1) );
        CHECK_CLOSE(  5.1, mp.getMaterialDensity(cell,1), 1e-5 );
        CHECK_EQUAL(  8, mp.getMaterialID(cell,2) );
        CHECK_CLOSE(  8.1, mp.getMaterialDensity(cell,2), 1e-5 );
        CHECK_EQUAL(  11, mp.getMaterialID(cell,3) );
        CHECK_CLOSE(  11.1, mp.getMaterialDensity(cell,3), 1e-5 );
        CHECK_EQUAL(  14, mp.getMaterialID(cell,4) );
        CHECK_CLOSE(  14.1, mp.getMaterialDensity(cell,4), 1e-5 );
    }

#if false
    TEST( bytesize_empty ) {
        MaterialProperties mp;
        CHECK_EQUAL( 56, sizeof( mp ) );
        CHECK_EQUAL( 72, mp.bytesize() );
        CHECK_EQUAL( 72, mp.capacitySize() );
    }
    TEST( bytesize_with_empty_cell ) {
        MaterialProperties mp;
        mp.add();
        CHECK_EQUAL( 56, sizeof( mp ) );
        CHECK_EQUAL( 104, mp.bytesize() );
        CHECK_EQUAL( 104, mp.capacitySize() );
    }
    TEST( bytesize_with_two_cells ) {
         MaterialProperties mp;
         mp.add();
         mp.add();
         CHECK_EQUAL( 136, mp.bytesize() );
         CHECK_EQUAL( 136, mp.capacitySize() );
     }
    TEST( bytesize_with_three_cells ) {
         MaterialProperties mp;
         mp.add();
         mp.add();
         mp.add();
         CHECK_EQUAL( 168, mp.bytesize() );
         CHECK_EQUAL( 200, mp.capacitySize() );
     }
    TEST( bytesize_with_four_cells ) {
        MaterialProperties mp;
        mp.add();
        mp.add();
        mp.add();
        mp.add();
        CHECK_EQUAL( 200, mp.bytesize() );
        CHECK_EQUAL( 200, mp.capacitySize() );
    }

    // Tests for large mesh sizes
    // Allocates up to 1GB so may break some concurrent builds
    TEST( bytesize_of_1million_empty_cell ) {
        MaterialProperties mp;
        for( unsigned i=0; i<1e6; ++i ) {
            mp.add();
        }
        CHECK_EQUAL( 32000072, mp.bytesize() );
        CHECK_EQUAL( 33554504, mp.capacitySize() );
    }

    TEST( bytesize_of_1million_cells_with_34_materials ) {
        MaterialProperties mp;

        for( unsigned i=0; i<1e6; ++i ) {
            mp.add();
            for( unsigned j=0; j<34; ++j ) {
                mp.addCellMaterial( i, j, double(j)+0.1);
            }
        }

        CHECK_EQUAL( 1e6, mp.size() );
        CHECK_EQUAL( 1048576, mp.capacity());
        CHECK_EQUAL( 576000072, mp.bytesize() ); // 576 MB
        CHECK_EQUAL( 805896264, mp.capacitySize() ); // 806 MB
        CHECK_EQUAL( 14271360, mp.numEmptyMatSpecs() );
    }

    TEST( bytesize_with_fixed_1million_cells_with_34_materials ) {
        MaterialProperties mp(1e6);
        CHECK_EQUAL( 72, sizeof( mp ) );

        for( unsigned i=0; i<1e6; ++i ) {
            for( unsigned j=0; j<34; ++j ) {
                mp.addCellMaterial( i, j, double(j)+0.1);
            }
        }
        CHECK_EQUAL( 1e6, mp.size() );
        CHECK_EQUAL( 1e6, mp.capacity());
        CHECK_EQUAL(  576000072, mp.bytesize() ); // 576 MB
        CHECK_EQUAL( 1056000072, mp.capacitySize() ); // 1056 MB
        CHECK_EQUAL( 30000000, mp.numEmptyMatSpecs() );
    }


    TEST( bytesize_with_fixed_1million_cells_add_cell_with_shrink_to_fit ) {
        MaterialProperties mp;
        CHECK_EQUAL( 72, sizeof( mp ) );

        mcatk::CellProperties cell;
        for( unsigned j=0; j<34; ++j ) {
            cell.add(  j, double(j)+0.1 );
        }
        cell.shrink_to_fit();

        for( unsigned i=0; i<1e6; ++i ) {
            mp.add(cell);
        }
        CHECK_EQUAL( 1e6, mp.size() );
        CHECK_EQUAL( 1048576, mp.capacity());
        CHECK_EQUAL( 576000072, mp.bytesize() ); // 576 MB
        CHECK_EQUAL( 577554504, mp.capacitySize() ); // 1056 MB
        CHECK_EQUAL( 0, mp.numEmptyMatSpecs() );
    }
#endif



}
} // end namespace
