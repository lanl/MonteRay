#include "NextEventEstimator.hh"
#include <iostream>
#include <limits>

namespace MonteRay{

void NextEventEstimator::printPointDets(const std::string& outputFile, int nSamples, int constantDimension) const {
  if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

  if( tallyPoints_.size() == 0 ) {
      return;
  }

  std::ofstream out;
  out.open(outputFile.c_str(), std::ios::out );
  if( ! out.is_open() ) {
      throw std::runtime_error( "Failure opening output file.  File= " + outputFile );
  }
  outputTimeBinnedTotal( out, nSamples, constantDimension);
  out.close();
}

void NextEventEstimator::outputTimeBinnedTotal(std::ostream& out, int nSamples, int constantDimension) const {
  out << "#  MonteRay results                                                    \n";
  out << "#                                                                      \n";
  out << "#       X          Y          Z      Time        Score        Score    \n";
  out << "#     (cm)       (cm)       (cm)   (shakes)      Average      Rel Err. \n";
  out << "# ________   ________   ________   ________   ___________   ___________ \n";

  TallyFloat sum = 0.0;
  TallyFloat min = std::numeric_limits<double>::infinity();
  TallyFloat max = -std::numeric_limits<double>::infinity();

  // dim2 used to insert new-line when it decreases, indicating a new row.
  unsigned dim2;
  switch (constantDimension) {
  case 0:
      dim2 = 2; // z
      break;
  case 1:
      dim2 = 2; // z
      break;
  case 2:
      dim2 = 1; // y
      break;
  default:
      break;
  }

  const auto& pos = getPoint(0);

  // previousSecondDimPosition used to detect when to insert carriage return
  double previousSecondDimPosition = pos[dim2];

  for( int i=0; i < this->nSpatialBins(); i++) {
    for ( size_t j = 0; j < this->nEnergyBins(); j++) {
      for ( size_t k = 0; k < this->nTimeBins(); k++) {
        auto index = this->getIndex(i, j, k);
        auto& pos = getPoint(i);
        double energy = this->energyBinEdges()[j];
        double time = this->timeBinEdges()[j];
        auto mean = this->mean(index);
        auto stdDev = this->stdDev(index);

        if(  pos[dim2] < previousSecondDimPosition ) {
            out << "\n";
        }
        char buffer[200];
        snprintf( buffer, 200, "  %8.3f   %8.3f   %8.3f   %8.3f   %8.3f   %11.4e   %11.4e\n",
                                 pos[0], pos[1], pos[2],  energy,  time,   mean,    stdDev );
        out << buffer;

        previousSecondDimPosition = pos[dim2];

        if( mean < min ) min = mean;
        if( mean > max ) max = mean;
        sum += mean;
      }
    }
  }
  out << "\n#\n";
  out << "# Min mean = " << min << "\n";
  out << "# Max mean = " << max << "\n";
  out << "# Average mean = " << sum / tallyPoints_.size() << "\n";
}

}
