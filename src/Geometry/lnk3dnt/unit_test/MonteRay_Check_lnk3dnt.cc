#include "UnitTest++.h"

#include "MonteRay_ReadLnk3dnt.hh"
#include <stdexcept>

// Test that input specific to the lnk3dnt format is allowed.

SUITE( Check_lnk3dnt ) {
	 using namespace std;
	 using namespace MonteRay;

     TEST( FileContainsGoodAMR ) {
   	    try {
   		    MonteRay_ReadLnk3dnt reader( "2by2by2_box.lnk3dnt" );
   	        }
   	    catch(std::exception& error) {
   	    	CHECK(false);
            CHECK_EQUAL("ERROR:  LNK3DNT Reader will not read file version 5\n", error.what());
   	    }
     }
     TEST( FileContainsBadAMR ) {
    	 try {
    		 MonteRay_ReadLnk3dnt reader( "2by2by2_box_lev5.lnk3dnt" );
    		 reader.ReadMatData();
    		 CHECK(false);
    	 }
    	 catch(std::exception& error) {
    	     string msg( error.what() );
             CHECK( msg.find("LNK3DNT Reader will not read file version 5") != string::npos );
    	 }

     }
     TEST( FileContainsBadDensity ) {
    	 try {
    		 MonteRay_ReadLnk3dnt reader(  "2by2by2_box_neg_density.lnk3dnt" );
             reader.ReadMatData();
    		 CHECK(false);
    	 }
    	 catch(std::exception& error) {
    		 string msg( error.what() );
             CHECK( msg.find("LNK3DNT File contains a negative density:-18.7") != string::npos );
    	 }
     }
     TEST( FileContainsBadMatID ) {
      	 try {
       		 MonteRay_ReadLnk3dnt reader(  "2by2by2_box_neg_matID.lnk3dnt" );
       		 reader.ReadMatData();
       		 CHECK(false);
      	 	}
      	 catch(std::exception& error) {
      		 string msg( error.what() );
      		 CHECK( msg.find("LNK3DNT File contains a negative or 0 MatId:-268435455") != string::npos );
      	 }

     }

}

