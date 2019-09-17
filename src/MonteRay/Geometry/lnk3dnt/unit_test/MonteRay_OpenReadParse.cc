#include "UnitTest++.h"

#include "MonteRay_ReadLnk3dnt.hh"
#include "MonteRay_BinaryReadFcns.hh"
#include "MonteRay_BinaryWriteFcns.hh"

#include <stdexcept>
#include <iostream>

namespace MonteRay_OpenReadParse_test {

using namespace std;
using namespace MonteRay;

typedef MonteRay_BinaryReader BinaryReader;
typedef MonteRay_BinaryWriter BinaryWriter;


SUITE( BinaryReadingFcns ) {

    const char filename[] = "test_data";

    TEST( FailToOpenInput ) {
        CHECK_THROW( BinaryReader( "somename" ), std::exception );
    }
    TEST( ReadSingleDouble ) {
        double WriteValue = 1.23e-24;
        BinaryWriter file( filename );
        file.append( WriteValue );

        double ReadValue;
        BinaryReader in( filename );
        in.read( ReadValue );

        CHECK_EQUAL( WriteValue, ReadValue );
    }
    TEST( ReadWriteDouble ) {
        const char* filename = "dblsize";
        double WriteValue = 1000.0;
        BinaryWriter file( filename );
        file.append( WriteValue );

        double ReadValue;
        BinaryReader in( filename );
        in.read( ReadValue );
        CHECK_EQUAL( WriteValue, ReadValue );

    }
    TEST( ReadWriteInt ) {
        const char* filename = "intsize";
        int WriteValue = 1000;
        BinaryWriter file( filename );
        file.append( WriteValue );

        int ReadValue;
        BinaryReader in( filename );
        in.read( ReadValue );
        CHECK_EQUAL( WriteValue, ReadValue );

    }
    TEST( ReadWriteString ) {
        const char* filename = "stringsize";
        string WriteValue( "filename thats too long" );
        BinaryWriter file( filename );
        string ShortenedString( WriteValue, 0, 8 );
        file.append( WriteValue, 8 );

        string ReadValue;
        BinaryReader in( filename );
        in.read( ReadValue, 8 );
        CHECK_EQUAL( ShortenedString, ReadValue );

    }
    TEST( ReadSingleInt ) {
        int WriteValue = 9381023;
        BinaryWriter file( filename );
        file.append( WriteValue );

        int ReadValue;
        BinaryReader infile( filename );
        infile.read( ReadValue );

        CHECK_EQUAL( WriteValue, ReadValue );
    }

    TEST( ReadManyDouble ) {
        const unsigned N = 15;
        vector<double> WriteValue(N, 1.23e-24);
        BinaryWriter file( filename );
        file.append( WriteValue );

        vector<double> ReadValue;
        BinaryReader infile( filename );
        infile.read( ReadValue, N );
        CHECK_EQUAL( N, ReadValue.size() );
        CHECK_ARRAY_CLOSE( WriteValue, ReadValue, N, 1.0e-14 );
    }

    TEST( ReadManyInts ) {
        const unsigned N = 15;
        int WriteValue[ N ] = { 1, 3, -4, 9, 9999, 0, -1 };
        BinaryWriter file( filename );
        file.append( WriteValue );

        vector<int> ReadValue;
        BinaryReader infile( filename );
        infile.read( ReadValue, N );

        CHECK_EQUAL( N, ReadValue.size() );
        CHECK_ARRAY_CLOSE( WriteValue, ReadValue, N, 0 );
    }

    TEST( Simple_SetMark ) {
            const unsigned N = 15;
            const unsigned N2 = 25;
            unsigned val,val2;
            BinaryWriter outfile( filename );
            BinaryReader infile( filename );
            outfile.append( N );
            outfile.append( N2 );
            outfile.append( N+N2 );
            infile.setMark("start");
            infile.read(val);
            infile.read(val2);
            CHECK(val != val2 );
            infile.useMark("start");
		    infile.read(val2);
		    CHECK_EQUAL(val, val2 );
        }
    TEST( Complex_SetMark ) {
    	BinaryReader infile( filename );
    	unsigned val,val2,val3;
    	infile.read(val);
    	infile.setMark("oneval");
		infile.read(val2);
    	infile.setMark("twoval");
		infile.read(val3);
		infile.setMark("end");
		infile.useMark("oneval");
		infile.read(val3);
		CHECK_EQUAL(val3,val2);
    }
    TEST( Invalid_Read ) {
        	BinaryReader infile( filename );
        	unsigned val,val2,val3;
    		infile.read(val3);
    		infile.read(val3);
    		infile.read(val3);
    	    CHECK_THROW(infile.read(val3),std::logic_error);
        }


}

SUITE( ReadingMeta) {
	typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;

    const char filename[] = "test_data";

    TEST( Read_Meta) {
      BinaryWriter file( filename );

      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( 2   ); //FileVersion
      file.append( 14  ); //GeometrySpec
      file.append( 1000); //NZones
      file.append( -99 ); //EmptyInt
      file.append( -99 ); //EmptyInt

      const unsigned dim = 3;
      const unsigned NCellsCoarse = 5;
      const unsigned NCellsPerCoarse = 2;
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse ); //NCellsCoarse
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse * NCellsPerCoarse ); //NCellsFine

      for (int i=0; i<13; ++i)
        file.append( -99 );  //Empty

      file.append( 6   ); //MaxNMatsInCell
      file.append( -99 ); //Empty
      file.append( -99 ); //Empty
      file.append( 2   ); //Block Levels

      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              double vertValue = r * 1.0;
              file.append( vertValue );
          }
          file.append( 10.0 );
      }
      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              file.append( NCellsPerCoarse );
          }
      }


      ReadLnk3dnt  reader( filename );

      //Need access functions now ?

      int FileVersion=reader.getFileVersion();
      CHECK_EQUAL( 2 , FileVersion);

      int GeometrySpec=reader.getGeometrySpec();
      CHECK_EQUAL( 14 , GeometrySpec);

      int BlockLevels = reader.getBlockLevels();
      CHECK_EQUAL( 2 , BlockLevels);

    }
    TEST( Read_Meta_failVertCheck ) {
      BinaryWriter file( filename );

      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( 2   ); //FileVersion
      file.append( 14  ); //GeometrySpec
      file.append( 1000); //NZones
      file.append( -99 ); //EmptyInt
      file.append( -99 ); //EmptyInt

      const unsigned dim = 3;
      const unsigned NCellsCoarse = 5;
      const unsigned NCellsPerCoarse = 2;
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse ); //NCellsCoarse
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse * NCellsPerCoarse ); //NCellsFine

      for (int i=0; i<13; ++i)
        file.append( -99 );  //Empty

      file.append( 6   ); //MaxNMatsInCell
      file.append( -99 ); //Empty
      file.append( -99 ); //Empty
      file.append( 2   ); //Block Levels

      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              double vertValue = 1.0;
              file.append( vertValue );
          }
          file.append( 10.0 );
      }
      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              file.append( NCellsPerCoarse );
          }
      }

      CHECK_THROW( ReadLnk3dnt r( filename ), std::runtime_error );
    }
    TEST( Read_Meta_failNCellCheck ) {
      BinaryWriter file( filename );

      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( 2   ); //FileVersion
      file.append( 14  ); //GeometrySpec
      file.append( 1000); //NZones
      file.append( -99 ); //EmptyInt
      file.append( -99 ); //EmptyInt

      const unsigned dim = 3;
      const unsigned NCellsCoarse = 5;
      const unsigned NCellsPerCoarse = 2;
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse ); //NCellsCoarse
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse * NCellsPerCoarse ); //NCellsFine

      for (int i=0; i<13; ++i)
        file.append( -99 );  //Empty

      file.append( 6   ); //MaxNMatsInCell
      file.append( -99 ); //Empty
      file.append( -99 ); //Empty
      file.append( 2   ); //Block Levels

      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              double vertValue = r * 1.0;
              file.append( vertValue );
          }
//          file.append( 10.0 ); // this makes the #verts wrong!
      }
      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              file.append( NCellsPerCoarse );
          }
      }

      CHECK_THROW( ReadLnk3dnt r( filename ), std::runtime_error );
    }
    TEST( Read_Meta_failNCellnotCorrect ) {
      BinaryWriter file( filename );

      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( string("filename"), 8 ); //Empty
      file.append( 2   ); //FileVersion
      file.append( 14  ); //GeometrySpec
      file.append( 1000); //NZones
      file.append( -99 ); //EmptyInt
      file.append( -99 ); //EmptyInt

      const unsigned dim = 3;
      const unsigned NCellsCoarse = 5;
      const unsigned NCellsPerCoarse = 2;
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse ); //NCellsCoarse
      for( unsigned d=0; d<dim; ++d )
          file.append( NCellsCoarse * NCellsPerCoarse ); //NCellsFine

      for (int i=0; i<13; ++i)
        file.append( -99 );  //Empty

      file.append( 6   ); //MaxNMatsInCell
      file.append( -99 ); //Empty
      file.append( -99 ); //Empty
      file.append( 2   ); //Block Levels

      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              double vertValue = r * 1.0;
              file.append( vertValue );
          }
          file.append( 10.0 ); // this makes the #verts wrong!
      }
      for( unsigned d=0; d<dim; ++d ) {
          for( unsigned r=0; r<NCellsCoarse; ++r ) {
              file.append( NCellsPerCoarse+1 ); // should just be NCellsPerCoarse
          }
      }

      CHECK_THROW( ReadLnk3dnt r( filename ), std::runtime_error );
    }
}
SUITE( ReadingLnk3dnt ) {
	typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;

    TEST( CantFindLinkFile ) {
        CHECK_THROW( ReadLnk3dnt("NotAFile"), std::exception );
    }
    TEST( ReadingLinkFile ) {
        ReadLnk3dnt reader( "lnk3dnt/3shells.lnk3dnt" );
        reader.ReadMatData();
        CHECK( true );
    }
    TEST( ReadingGodivaLinkFile ) {
        ReadLnk3dnt reader( "lnk3dnt/godiva.lnk3dnt" );
        reader.ReadMatData();
  //      reader.Print( cout );
        CHECK( true );
    }
}

SUITE(ReadingSphericalLnk3dnt ) {
	typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;

    TEST( ReadingGodivaSpherical ) {
        ReadLnk3dnt reader( "lnk3dnt/godiva_spherical.lnk3dnt" );
        reader.ReadMatData();
//        reader.Print( cout );
        CHECK( true );
    }
}
SUITE(ReadingCylindricalRZTLnk3dnt ) {
	typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;

    TEST( ReadingGodivaCylindrical ) {
        string file = "lnk3dnt/hfm015rzt.linkout";

        ReadLnk3dnt reader( file );
        reader.ReadMatData();
        //reader.Print( cout );
        CHECK( true );
    }
}

SUITE(ReadingBadPath ) {
	typedef MonteRay_ReadLnk3dnt ReadLnk3dnt;

    TEST( ReadingBadPath_BadLocalFile ) {
        CHECK_THROW( ReadLnk3dnt( "/BadPath/godiva_spherical.bad" ), std::exception );
    }
//    TEST( ReadingBadPath_GoodLocalFile ) {
//        ReadLnk3dnt( "/BadPath/SuperSmall.lnk3dnt" );
//        CHECK( true );
//    }
}

} // end namespace
