#include <UnitTest++.h>

#include "MonteRay_MaterialProperties_FlatLayout.hh"
#include "MonteRay_CellProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"

#include <iostream>
#include <vector>
#include <cmath>

#include <cassert>

namespace MaterialProperties_FlatLayout_tester{
using namespace std;
using namespace MonteRay;

typedef MonteRay_CellProperties CellProperties;
typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;
typedef MonteRay_MaterialSpec MaterialSpec;
typedef MonteRay_MaterialProperties_FlatLayout MaterialProperties_FlatLayout;

SUITE( MaterialProperties_FlatLayout_tests ) {
    typedef CellProperties CellProperty_t;

    double epsilon = 1e-13;

    class tester : public MaterialProperties_FlatLayout {
    public:
        tester() : MaterialProperties_FlatLayout(){}
        ~tester(){}

        void testconvertFromSingleNumComponents(){
            convertFromSingleNumComponents();
        }
    };

    TEST(default_single_temperature) {
        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL(true, mp.isSingleTemp() );
        CHECK_EQUAL(0, mp.temperatureSize() );
    }

    TEST(add_cell_default_temp) {
        CellProperty_t cell;
        cell.add( 1, 0.1);
        cell.add( 2, 0.2);

        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell );
        CHECK_EQUAL( 1, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_CLOSE( -99.0      , mp.getTemperature(0), 1e-7 );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);
        CHECK_EQUAL(true, mp.isSingleTemp() );
        CHECK_EQUAL(1, mp.temperatureSize() );
    }

    TEST(add_cell_temp) {
        CellProperty_t cell;
        cell.add( 1, 0.1);
        cell.add( 2, 0.2);
        cell.setTemperature( 11.0 );

        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell );
        CHECK_EQUAL( 1, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_CLOSE( 11.0      , mp.getTemperature(0), 1e-7 );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);
        CHECK_EQUAL(true, mp.isSingleTemp() );
        CHECK_EQUAL(1, mp.temperatureSize() );
    }

    TEST(add_two_cells) {
        CellProperty_t cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);

        CellProperty_t cell2;
        cell2.add( 21, 2.1);
        cell2.add( 22, 2.2);
        cell2.add( 23, 2.3);

        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell1 );
        mp.add( cell2 );
        CHECK_EQUAL( 2, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( -99.0     , mp.getTemperature(0), 1e-7 );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);

        CHECK_EQUAL( 3         , mp.getNumMaterials(1) );
        CHECK_CLOSE( -99.0     , mp.getTemperature(1), 1e-7 );
        CHECK_EQUAL( 23         , mp.getMaterialID(1,2) );
        CHECK_CLOSE( 2.3        , mp.getMaterialDensity(1,2), 1e-04);
        CHECK_EQUAL(true, mp.isSingleTemp() );
        CHECK_EQUAL(1, mp.temperatureSize() );
    }

    TEST(add_two_cells_different_temps) {
        CellProperty_t cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);
        cell1.setTemperature( 11.0 );

        CellProperty_t cell2;
        cell2.add( 21, 2.1);
        cell2.add( 22, 2.2);
        cell2.add( 23, 2.3);
        cell2.setTemperature( 21.0 );

        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell1 );
        mp.add( cell2 );
        CHECK_EQUAL( 2, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_CLOSE( 11.0      , mp.getTemperature(0), 1e-7 );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);

        CHECK_EQUAL( 3         , mp.getNumMaterials(1) );
        CHECK_CLOSE( 21.0      , mp.getTemperature(1), 1e-7 );
        CHECK_EQUAL( 23         , mp.getMaterialID(1,2) );
        CHECK_CLOSE( 2.3       , mp.getMaterialDensity(1,2), 1e-04);
        CHECK_EQUAL(false, mp.isSingleTemp() );
        CHECK_EQUAL(2, mp.temperatureSize() );
    }

    TEST(add_single_cell_disable_reduction) {
        CellProperty_t cell;
        cell.add( 1, 0.1);
        cell.add( 2, 0.2);
        cell.setTemperature( 11.0 );

        MaterialProperties_FlatLayout mp;
        mp.disableMemoryReduction();

        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell );
        CHECK_EQUAL( 1, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_CLOSE( 11.0      , mp.getTemperature(0), 1e-7 );
        CHECK_EQUAL( 2         , mp.getMaterialID(0,1) );
        CHECK_CLOSE( 0.2       , mp.getMaterialDensity(0,1), 1e-04);
        CHECK_EQUAL(false, mp.isSingleTemp() );
        CHECK_EQUAL(false, mp.isSingleNumMats() );
        CHECK_EQUAL(1, mp.temperatureSize() );
    }

    TEST(add_two_cell_disable_reduction) {
        CellProperty_t cell;
        cell.add( 1, 0.1);
        cell.add( 2, 0.2);
        cell.setTemperature( 11.0 );

        MaterialProperties_FlatLayout mp;
        mp.disableMemoryReduction();

        CHECK_EQUAL( 0, mp.size() );
        mp.add( cell );
        mp.add( cell );
        CHECK_EQUAL( 2, mp.size() );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_EQUAL( 2         , mp.getNumMaterials(0) );
        CHECK_EQUAL(false, mp.isSingleTemp() );
        CHECK_EQUAL(false, mp.isSingleNumMats() );
        CHECK_EQUAL(2, mp.temperatureSize() );
    }

    TEST(initializeMaterialDescription){

        MaterialProperties_FlatLayout mp;

        vector<MaterialProperties_FlatLayout::MatID_t> IDs;
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

        mp.initializeMaterialDescription( IDs, densities, TotalNumberOfCells);

        int cellNumber = 3;
        CHECK_EQUAL( size_t(1) , mp.getNumMaterials(cellNumber) );
        CHECK_EQUAL( 2         , mp.getMaterialID(cellNumber,0) );
        CHECK_CLOSE( 2.0       , mp.getMaterialDensity(cellNumber,0), 1e-04);
    }

    TEST(Ctor_withLnk3dnt){

        ReadLnk3dnt readerObject( "lnk3dnt/3iso_3shell_godiva.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties_FlatLayout problemMaterialProperties;
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

    TEST( clear ){
        vector<MaterialProperties_FlatLayout::MatID_t> IDs;
        IDs.push_back(1);
        IDs.push_back(1);

        vector<double> densities;
        densities.push_back(1.0);
        densities.push_back(1.0);

        size_t TotalNumberOfCells = 2;
        MaterialProperties_FlatLayout mp( IDs, densities, TotalNumberOfCells );

        CHECK_EQUAL( 2, mp.size() );
        mp.clear();
        CHECK_EQUAL( 0, mp.size() );
    }

    class TestParticle_t {
    public:
        TestParticle_t(){
            cellIndex = -2;
        };
        ~TestParticle_t(){};

        int cellIndex;
        void setLocationIndex( int index) { cellIndex = index; }
        MaterialProperties_FlatLayout::Cell_Index_t getLocationIndex(void) const { return cellIndex; }
    };

    TEST(getNumMaterial_from_Particle_and_index){
        CellProperty_t cell1;
        cell1.add( 1, 0.1);
        cell1.add( 2, 0.2);

        CellProperty_t cell2;
        cell2.add( 1, 0.1);
        cell2.add( 2, 0.2);
        cell2.add( 3, 0.3);

        MaterialProperties_FlatLayout mp;
        mp.add(cell1);
        mp.add(cell2);

        TestParticle_t p;
        p.setLocationIndex( 0 );

        CHECK_EQUAL( 2, mp.getNumMaterials(0) );
     }

    TEST(SetGlobalTemperature){

        vector<MaterialProperties_FlatLayout::MatID_t> IDs;
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
        MaterialProperties_FlatLayout MP1( IDs, densities, TotalNumberOfCells );
        MP1.setGlobalTemperature( 0.001 );

        CHECK_CLOSE(0.001 , MP1.getTemperature(3), epsilon );
    }

    TEST(SetIndividualTemperature){

         vector<MaterialProperties_FlatLayout::MatID_t> IDs;
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
         MaterialProperties_FlatLayout MP1( IDs, densities, TotalNumberOfCells );
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

     }

    TEST(SetCellTemperatures){

         vector<MaterialProperties_FlatLayout::MatID_t> IDs;
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
         MaterialProperties_FlatLayout MP1( IDs, densities, TotalNumberOfCells );

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

         vector<MaterialProperties_FlatLayout::MatID_t> IDs;
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
         MaterialProperties_FlatLayout MP1( IDs, densities, TotalNumberOfCells );

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

    TEST(load_with_singleMat_and_expand){

        vector<MaterialProperties_FlatLayout::MatID_t> IDs;
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(3);
        IDs.push_back(4);
        IDs.push_back(5);

        vector<double> densities;
        densities.push_back(1.1);
        densities.push_back(2.1);
        densities.push_back(3.1);
        densities.push_back(4.1);
        densities.push_back(5.1);

        size_t TotalNumberOfCells = 5;
        tester mp;
        mp.initializeMaterialDescription( IDs, densities, TotalNumberOfCells );
        CHECK_EQUAL( true, mp.isSingleNumMats());
        CHECK_EQUAL(1, mp.getNumMaterials(0));
        CHECK_EQUAL(1, mp.getMaterialID(0,0));
        CHECK_CLOSE(1.1, mp.getMaterialDensity(0,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(1));
        CHECK_EQUAL(2, mp.getMaterialID(1,0));
        CHECK_CLOSE(2.1, mp.getMaterialDensity(1,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(2));
        CHECK_EQUAL(3, mp.getMaterialID(2,0));
        CHECK_CLOSE(3.1, mp.getMaterialDensity(2,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(3));
        CHECK_EQUAL(4, mp.getMaterialID(3,0));
        CHECK_CLOSE(4.1, mp.getMaterialDensity(3,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(4));
        CHECK_EQUAL(5, mp.getMaterialID(4,0));
        CHECK_CLOSE(5.1, mp.getMaterialDensity(4,0), 1e-7);

        mp.testconvertFromSingleNumComponents();
        CHECK_EQUAL( false, mp.isSingleNumMats());

        CHECK_EQUAL(1, mp.getNumMaterials(0));
        CHECK_EQUAL(1, mp.getMaterialID(0,0));
        CHECK_CLOSE(1.1, mp.getMaterialDensity(0,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(1));
        CHECK_EQUAL(2, mp.getMaterialID(1,0));
        CHECK_CLOSE(2.1, mp.getMaterialDensity(1,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(2));
        CHECK_EQUAL(3, mp.getMaterialID(2,0));
        CHECK_CLOSE(3.1, mp.getMaterialDensity(2,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(3));
        CHECK_EQUAL(4, mp.getMaterialID(3,0));
        CHECK_CLOSE(4.1, mp.getMaterialDensity(3,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(4));
        CHECK_EQUAL(5, mp.getMaterialID(4,0));
        CHECK_CLOSE(5.1, mp.getMaterialDensity(4,0), 1e-7);
    }

    TEST(load_with_singleMat_disableReduction){

        vector<MaterialProperties_FlatLayout::MatID_t> IDs;
        IDs.push_back(1);
        IDs.push_back(2);
        IDs.push_back(3);
        IDs.push_back(4);
        IDs.push_back(5);

        vector<double> densities;
        densities.push_back(1.1);
        densities.push_back(2.1);
        densities.push_back(3.1);
        densities.push_back(4.1);
        densities.push_back(5.1);

        size_t TotalNumberOfCells = 5;
        tester mp;
        mp.disableMemoryReduction();
        mp.initializeMaterialDescription( IDs, densities, TotalNumberOfCells );
        CHECK_EQUAL( false, mp.isSingleNumMats());

        CHECK_EQUAL(1, mp.getNumMaterials(0));
        CHECK_EQUAL(1, mp.getMaterialID(0,0));
        CHECK_CLOSE(1.1, mp.getMaterialDensity(0,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(1));
        CHECK_EQUAL(2, mp.getMaterialID(1,0));
        CHECK_CLOSE(2.1, mp.getMaterialDensity(1,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(2));
        CHECK_EQUAL(3, mp.getMaterialID(2,0));
        CHECK_CLOSE(3.1, mp.getMaterialDensity(2,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(3));
        CHECK_EQUAL(4, mp.getMaterialID(3,0));
        CHECK_CLOSE(4.1, mp.getMaterialDensity(3,0), 1e-7);

        CHECK_EQUAL(1, mp.getNumMaterials(4));
        CHECK_EQUAL(5, mp.getMaterialID(4,0));
        CHECK_CLOSE(5.1, mp.getMaterialDensity(4,0), 1e-7);

        CHECK_EQUAL(5, mp.temperatureSize() );
    }

    TEST(AddCellMaterial){

        const int NumberOfCells = 20;
        std::vector<MaterialProperties_FlatLayout::MatID_t> IDs( NumberOfCells, 99);
        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties_FlatLayout MP( IDs, densities, NumberOfCells );

        const unsigned cellID = 5;
        CHECK_EQUAL( 1, MP.getNumMaterials(cellID));
        CHECK_EQUAL( IDs[cellID]         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID]   , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MP.addCellMaterial(cellID, IDnew, densityNew );

        CHECK_EQUAL(size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( 2, MP.getNumMaterials(cellID));
        CHECK_EQUAL( IDs[cellID]        , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID]   , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        CHECK_EQUAL( 1, MP.getNumMaterials(cellID+1));
        CHECK_EQUAL( IDs[cellID]        , MP.getMaterialID(cellID+1,0) );
        CHECK_CLOSE( densities[cellID]   , MP.getMaterialDensity(cellID+1,0), 1e-05 );
#ifdef BOOST_ENABLE_ASSERT_HANDLE
        CHECK_THROW( MP.getMaterialID(cellID+1,1), std::exception );
#endif
    }


    TEST( RemoveCellMaterial ){

        const int NumberOfCells = 20;
        std::vector<MaterialProperties_FlatLayout::MatID_t> IDs( NumberOfCells, 99);
        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties_FlatLayout MP( IDs, densities, NumberOfCells );

        const unsigned cellID = 5;
        CHECK_EQUAL( IDs[cellID]         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID]   , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MP.addCellMaterial(cellID, IDnew, densityNew );

        CHECK_EQUAL( size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( size_t(2), MP.getNumMaterials(cellID) );

        CHECK_EQUAL( IDs[cellID], MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID], MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        MP.removeMaterial(cellID, IDs[cellID] );
        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        CHECK_EQUAL( size_t(1) , MP.getNumMaterials(cellID) );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,0), 1e-05 );
   }

    TEST( ReplaceCellMaterial_via_remove_and_add ){

        const int NumberOfCells = 20;
        std::vector<MaterialProperties_FlatLayout::MatID_t> IDs( NumberOfCells, 99);
        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties_FlatLayout MP( IDs, densities, NumberOfCells );

        const unsigned cellID = 5;
        CHECK_EQUAL( IDs[cellID]         , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID]   , MP.getMaterialDensity(cellID,0), 1e-05 );

        const int IDnew = 9;
        const double densityNew = 75.0;
        MP.addCellMaterial(cellID, IDnew, densityNew );

        CHECK_EQUAL(size_t(NumberOfCells), MP.size());

        CHECK_EQUAL( size_t(2), MP.getNumMaterials(cellID) );
        CHECK_EQUAL( IDs[cellID], MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densities[cellID], MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDnew      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityNew , MP.getMaterialDensity(cellID,1), 1e-05 );

        const int    IDreplace      = 45;
        const double densityReplace = 55.0;

        MP.removeMaterial(cellID, IDs[cellID]);
        MP.addCellMaterial(cellID, IDreplace, densityReplace );

        CHECK_EQUAL(size_t(NumberOfCells), MP.size());
        CHECK_EQUAL( size_t(2),       MP.getNumMaterials(cellID) );
        CHECK_EQUAL( IDnew          , MP.getMaterialID(cellID,0) );
        CHECK_CLOSE( densityNew     , MP.getMaterialDensity(cellID,0), 1e-05 );
        CHECK_EQUAL( IDreplace      , MP.getMaterialID(cellID,1) );
        CHECK_CLOSE( densityReplace , MP.getMaterialDensity(cellID,1), 1e-05 );
    }

    TEST( ReNumberID_Directly ){
        const int NumberOfCells = 20;
        std::vector<MaterialProperties_FlatLayout::MatID_t> IDs( NumberOfCells, 99);
        IDs[0] = 50;

        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties_FlatLayout MP( IDs, densities, NumberOfCells );

        CHECK_EQUAL( 50, MP.getMaterialID(0,0) );
        CHECK_EQUAL( 99, MP.getMaterialID(1,0) );
        CHECK_EQUAL( 99, MP.getMaterialID(19,0) );

        MP.resetMaterialID(0, 0, -150);
        MP.resetMaterialID(19, 0, -199);

        CHECK_EQUAL( -150, MP.getMaterialID(0,0) );
        CHECK_EQUAL( 99, MP.getMaterialID(1,0) );
        CHECK_EQUAL( -199, MP.getMaterialID(19,0) );
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
        std::vector<MaterialProperties_FlatLayout::MatID_t> IDs( NumberOfCells, 99);
        IDs[0] = 50;

        std::vector<double> densities(NumberOfCells, 25.0);
        MaterialProperties_FlatLayout MP( IDs, densities, NumberOfCells );

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

#if false
    TEST( bytesize_empty ) {
        MaterialProperties_FlatLayout mp;
        CHECK_EQUAL( 176, sizeof( mp ) );
        CHECK_EQUAL( 176, mp.bytesize() );
        CHECK_EQUAL( 176, mp.capacitySize() );
    }
    TEST( bytesize_with_empty_cell ) {
        MaterialProperties_FlatLayout mp;
        mp.add();
        CHECK_EQUAL( 176, sizeof( mp ) );
        CHECK_EQUAL( 194, mp.bytesize() );
        CHECK_EQUAL( 194, mp.capacitySize() );
    }
    TEST( bytesize_with_two_cells ) {
        MaterialProperties_FlatLayout mp;
         mp.add();
         mp.add();
         CHECK_EQUAL( 196, mp.bytesize() );
         CHECK_EQUAL( 196, mp.capacitySize() );
     }
    TEST( bytesize_with_three_cells ) {
        MaterialProperties_FlatLayout mp;
         mp.add();
         mp.add();
         mp.add();
         CHECK_EQUAL( 214, mp.bytesize() );
         CHECK_EQUAL( 224, mp.capacitySize() );
     }
    TEST( bytesize_with_four_cells ) {
        MaterialProperties_FlatLayout mp;
        mp.add();
        mp.add();
        mp.add();
        mp.add();
        CHECK_EQUAL( 224, mp.bytesize() );
        CHECK_EQUAL( 224, mp.capacitySize() );
    }
#endif

#if false
    // Tests for large mesh sizes
    // Allocates up to 1GB so may break some concurrent builds
    TEST( bytesize_of_1million_empty_cell ) {
        MaterialProperties_FlatLayout mp;
        for( unsigned i=0; i<1e6; ++i ) {
            mp.add();
        }
        CHECK_EQUAL( 20000160, mp.bytesize() );
        CHECK_EQUAL( 20971680, mp.capacitySize() );
    }

    TEST( bytesize_of_1million_cells_with_34_materials_reserve_Cells_only ) {
        MaterialProperties_FlatLayout mp;
        mp.reserve( 1e6, 1e6);
        CHECK_EQUAL( 160, sizeof( mp ) );

        for( unsigned i=0; i<1e6; ++i ) {
            mp.add();
            for( unsigned j=0; j<34; ++j ) {
                mp.addCellMaterial( i, j, double(j)+0.1);
            }
        }
        CHECK_EQUAL( 1e6, mp.size() );
        CHECK_EQUAL( 1e6, mp.capacity());
        CHECK_EQUAL( 428000160, mp.bytesize() ); // 428 MB
        CHECK_EQUAL( 788000160, mp.capacitySize() ); // 826 MB
        CHECK_EQUAL( 30000000, mp.numEmptyMatSpecs() );
    }

    TEST( bytesize_of_1million_cells_with_34_materials_reserve ) {

        unsigned nMats = 34;

        MaterialProperties_FlatLayout mp;
        mp.reserve( 1e6, 1e6*nMats );

        for( unsigned i=0; i<1e6; ++i ) {
            mp.add();
            for( unsigned j=0; j<nMats; ++j ) {
                mp.addCellMaterial( i, j, double(j)+0.1);
            }
        }

        CHECK_EQUAL( 1e6, mp.size() );
        CHECK_EQUAL( 1e6, mp.capacity());
        CHECK_EQUAL(  428000160, mp.bytesize() ); // 428 MB
        CHECK_EQUAL(  428000160, mp.capacitySize() ); // 1056 MB
        CHECK_EQUAL( 0, mp.numEmptyMatSpecs() );
    }
#endif

#if false
    TEST( bytesize_of_1million_cells_with_50_materials_loop ) {

        unsigned nMats = 50;

        for( unsigned k = 0 ; k < nMats+1; ++k){

            MaterialProperties_FlatLayout mp;
            mp.disableMemoryReduction();
            mp.reserve( 1e6, 1e6*k );

            for( unsigned i=0; i<1e6; ++i ) {
                mp.add();
                for( unsigned j=0; j<k; ++j ) {
                    mp.addCellMaterial( i, j, double(j)+0.1);
                }
            }
            std::cout << "Debug:  nMats= " << k << " bytes= " << mp.bytesize() << "\n";
        }
    }
#endif

}


SUITE( MaterialProperties_FlatLayout_1million_godiva) {
    TEST(1millon_godiva_noReduction){

        ReadLnk3dnt readerObject( "lnk3dnt/godiva_1mm_eighth.lnk" );
        readerObject.ReadMatData();

        MaterialProperties_FlatLayout mp;
        mp.disableMemoryReduction();
        mp.setMaterialDescription( readerObject );

        CHECK_EQUAL(false,mp.isSingleTemp() );
        CHECK_EQUAL(false,mp.isSingleNumMats() );
        CHECK_EQUAL(1e6, mp.size());
        CHECK_EQUAL(18148446, mp.bytesize()); // 19.6MB
        CHECK_EQUAL(18148446, mp.capacitySize()); // 19.6MB

        unsigned cell= 1e5;
        CHECK_EQUAL(1, mp.getMaxNumMats() );
        CHECK_EQUAL(1, mp.getNumMaterials(cell) );
        CHECK_EQUAL(2, mp.getMaterialID(cell,0) );
    }

    TEST(1millon_godiva){

        ReadLnk3dnt readerObject( "lnk3dnt/godiva_1mm_eighth.lnk" );
        readerObject.ReadMatData();

        MaterialProperties_FlatLayout mp;
        mp.setMaterialDescription( readerObject );

        CHECK_EQUAL(true,mp.isSingleTemp() );
        CHECK_EQUAL(true,mp.isSingleNumMats() );
        CHECK_EQUAL(1e6, mp.size());
        CHECK_EQUAL(6000128, mp.bytesize()); // 10MB
        CHECK_EQUAL(6000128, mp.capacitySize()); // 10MB

        unsigned cell= 1e5;
        CHECK_EQUAL(1, mp.getMaxNumMats() );
        CHECK_EQUAL(1, mp.getNumMaterials(cell) );
        CHECK_EQUAL(2, mp.getMaterialID(cell,0) );
    }
}

SUITE( MaterialProperties_FlatLayout_boston) {
#if false
    TEST(boston_noReduction){

        ReadLnk3dnt readerObject( "lnk3dnt/boston.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties_FlatLayout mp;
        mp.disableMemoryReduction();
        mp.setMaterialDescription( readerObject );

        CHECK_EQUAL(99960000, mp.size());

        CHECK_EQUAL( mp.temperatureSize(), mp.temperatureCapacity());
        CHECK_EQUAL( mp.offsetSize(), mp.offsetCapacity());
        CHECK_EQUAL( mp.componentMatIDSize(), mp.componentMatIDCapacity());
        CHECK_EQUAL( mp.componentDensitySize(), mp.componentDensityCapacity());

        CHECK_EQUAL(2199120128, mp.bytesize());     // 2.6 GB
        CHECK_EQUAL(2199120128, mp.capacitySize());  // 2.6 GB
    }
    TEST(boston){

        ReadLnk3dnt readerObject( "lnk3dnt/boston.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties_FlatLayout mp;
        mp.setMaterialDescription( readerObject );

        CHECK_EQUAL(99960000, mp.size());

        CHECK_EQUAL(599760128, mp.bytesize());     // 1.2 GB
        CHECK_EQUAL(599760128, mp.capacitySize());  // 1.2 GB
    }
#endif
}


} // end namespace

