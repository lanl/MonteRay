#include "needebugger.hh"

#include <iostream>

#include "GPUUtilityFunctions.hh"
#include "GPUSync.hh"

#include "gpuTally.hh"

#include "MonteRay_timer.hh"
#include "RayListController.t.hh"
#include "GridBins.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.t.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "MonteRayNextEventEstimator.t.hh"
#include "MonteRayCrossSection.hh"

namespace nee_debugger_app {

using namespace MonteRay;

void
nee_debugger::checkFileExists(const std::string& filename){
    bool good = false;
    std::ifstream file(filename.c_str());
    good = file.good();

    if( ! good ) {
        std::cout << "nee_debugger::checkFileExists -- Can open filename = " << filename << std::endl;
        throw std::runtime_error("nee_debugger::checkFileExists -- can open file");
    }

    file.close();
}

void
nee_debugger::launch(const std::string& optBaseName){

    using Geom_t = MonteRay_SpatialGrid;
    //using Geom_t = GridBins;

    // next-event estimator
    // test nee save state file exists
    std::string baseName = optBaseName + std::string(".bin");

    std::string filename = std::string("nee_state_") + baseName;
    checkFileExists(filename);

    MonteRayNextEventEstimator<Geom_t> estimator(0);
    estimator.readFromFile( filename );

    // raylist
    filename = std::string("raylist_") + baseName;
    checkFileExists(filename);
    RayList_t<3> raylist(1);
    raylist.readFromFile( filename );

    // geometry
    filename = std::string("geometry_") + baseName;
    checkFileExists(filename);
    Geom_t grid;
    grid.readFromFile( filename );

    // material properties
    MonteRay_MaterialProperties matprops;
    filename = std::string("matProps_") + baseName;
    checkFileExists(filename);
    matprops.readFromFile( filename );

    // materials
    MonteRayMaterialListHost matlist(1);
    filename = std::string("materialList_") + baseName;
    checkFileExists(filename);
    matlist.readFromFile( filename );

    grid.copyToGPU();
    matlist.copyToGPU();
    matprops.copyToGPU();

    estimator.setGeometry( &grid, &matprops );
    estimator.setMaterialList( &matlist );
    estimator.copyToGPU();
    //estimator.dumpState( &raylist, "nee_debug_dump_test2" );

    for( unsigned i=0; i<raylist.size(); ++i ) {
        Ray_t<3> ray;
        ray = raylist.getParticle(i);

        RayList_t<3> singleSizeRaylist(1);
        singleSizeRaylist.add( ray );
        singleSizeRaylist.copyToGPU();

        RayWorkInfo rayInfo( singleSizeRaylist.size() );
        rayInfo.copyToGPU();

        GPUSync sync1; sync1.sync();

        std::cout << "Launching ray # " << i << std::endl;
        estimator.launch_ScoreRayList(-1,-1, &singleSizeRaylist, &rayInfo, 0, false);

        GPUSync sync2; sync2.sync();
    }

    estimator.copyToCPU();

    std::string outputFile = optBaseName + std::string("_tally.txt");
    estimator.printPointDets( outputFile, 1 );
}


} // end namespace
