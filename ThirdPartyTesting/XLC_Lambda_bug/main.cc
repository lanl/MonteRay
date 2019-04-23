#include <vector>
#include <map>
#include <functional>
#include <stdio.h>

std::map< int,double>
MassOfMaterial( const std::vector<double>& RegionVolumes ) {

    std::map< int,double> MaterialMass; // Table of masses, Captured by reference

    // Apply functor to every region
    for( unsigned region=0; region < RegionVolumes.size(); ++region ) {
        double currentVolume = RegionVolumes[ region ];

        // Computes the mass contribution from a single material within a region/cell

        // both variants fail with xlc++
        //auto massByRegion = [&MaterialMass, currentVolume] ( const int ID, const double den ) -> double {
        auto massByRegion = [&] (  int ID,  double den ) {
            double mass =  den * currentVolume;
            auto pair = MaterialMass.insert( std::make_pair(ID,mass) );
            bool worked = pair.second;
            auto iterator = pair.first;
            if( not worked ) { iterator->second += mass; }
            return mass;
        };

        massByRegion( 1, 10.0 );
    }

    // fails with XLC without the std::move
    return MaterialMass;
    //return std::move(MaterialMass);
}

int main(int argc, char** argv) {

    std::vector<double> volumes = { .9, 9.0, 90.0 };

    std::map< int,double> mass_of_each_material = MassOfMaterial( volumes );

    // 3 cells each with density 10.0,  .9*10 + 9*10 + 90*10 = 99.9*10 = 999
    double expected_value = 999.0;
    double value = mass_of_each_material[1];

    if( value < (expected_value-1.0e-14 ) or value > (expected_value+1.0e-14 ) ) {
        printf("FAILURE: value was %f but should be %f !! \n", value, expected_value );
    } else {
        printf("PASS: value is correct, %f .\n", value );
    }
    return 0;
}
