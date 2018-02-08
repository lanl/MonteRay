/*
 * MonteRay_TransportMeshTypeEnum.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_TRANSPORTMESHTYPEENUM_HH_
#define MONTERAY_TRANSPORTMESHTYPEENUM_HH_

namespace MonteRay {
    ///\brief A namespace to handle the mesh type enums for MonteRay.  Similar to MCATK
    namespace TransportMeshTypeEnum {
        enum TransportMeshTypeEnum_t{NONE=0, Cartesian, Cylindrical, Spherical, MAX};
    }
}


#endif /* MONTERAY_TRANSPORTMESHTYPEENUM_HH_ */
