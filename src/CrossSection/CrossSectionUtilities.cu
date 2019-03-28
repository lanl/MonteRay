#include "CrossSectionUtilities.hh"

#include <cmath>

namespace MonteRay {

void
thinGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error) {
    // thin grid
    bool done;
    do {
        done = true;
        unsigned i = 0;
        for( auto previous_itr = linearGrid.begin(); previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;
            auto next_itr = itr; ++next_itr;
            if( next_itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = next_itr->first;
            double energy = itr->first;

            // calculated interpolatedXS
            double lower =  previous_itr->second;
            double upper =  next_itr->second;
            double deltaE = energy2 - energy1;
            double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;

            // check difference with real xs
            double totalXS = xsFunc( energy );
            double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;
            //            printf( "Debug: i=%d  E=%f, interp=%f, real=%f diff=%f \n", i, energy, interpolatedXS, totalXS, percentDiff);
            if( percentDiff < max_error * 0.5 ) {
                linearGrid.erase(itr);
                done = false;
                break;
            }
            ++i;
        }
    } while( !done );
}

void
addPointsToGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error ) {
    bool done;
    // linearize
    do {
        done = true;
        for( auto previous_itr = linearGrid.begin(); previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = itr->first;
            double deltaE = energy2 - energy1;

            if( deltaE > 1e-6 ) {
                // don't add points finer than 1e-6
                double energy = std::exp(( std::log(energy2) - std::log(energy1) )*0.5 + std::log(energy1));

                // calculated interpolatedXS
                double lower =  previous_itr->second;
                double upper =  itr->second;

                double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;

                // check difference with real xs
                double totalXS = xsFunc( energy );
                double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;

                if( percentDiff > max_error ) {
                    linearGrid.insert(itr, std::make_pair(energy, totalXS));
                    done = false;
                }
            }
        }

    } while ( !done );
}

bool
checkGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error, unsigned nIntermediateBins){
    const bool debug = false;

    if( debug ) printf( "Debug: createLinearGrid - checking linearization\n");

    // check linearization
    bool done = true;
    do {
        done = true;
        auto start_itr = linearGrid.begin();
        for( auto previous_itr = start_itr; previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = itr->first;
            double deltaE = energy2 - energy1;

            double lower =  previous_itr->second;
            double upper =  itr->second;

            // no need to go below 1-eV for photon data
            if( std::abs( deltaE ) > 1e-6 ) {
                nIntermediateBins = std::min( unsigned( deltaE / 1e-6 ), nIntermediateBins ) ;
            } else {
                nIntermediateBins = 0;
            }

            for( auto j=0; j<nIntermediateBins; ++j) {
                double energy = energy1 + (deltaE*j)/nIntermediateBins;
                // calculated interpolatedXS
                double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;
                double totalXS = xsFunc( energy );
                double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;

                if( percentDiff > max_error ) {
                    if( debug ) {
                        printf( "Debug: createLinearGrid - linearization failed for E=%.10f, real XS=%f, interpolated XS=%f, percent diff=%f\n",
                                energy, totalXS, interpolatedXS, percentDiff );

                    }
                    start_itr = linearGrid.insert(itr, std::make_pair(energy, totalXS));
                    done = false;
                    break;
                }

                if( debug ) {
                    printf( "Debug: createLinearGrid - linearization passed for E=%.10f, real XS=%f, interpolated XS=%f, percent diff=%f\n",
                            energy, totalXS, interpolatedXS, percentDiff );
                }
            }

        }
    } while ( !done );
    return true;

}

}
