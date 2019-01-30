# Dashboard is opened for submissions for a 24 hour period starting at
# the specified NIGHLY_START_TIME. Time is specified in 24 hour format.
SET (CTEST_NIGHTLY_START_TIME "20:00:00 MDT")

set( CTEST_DROP_METHOD     "https" )
set( CTEST_DROP_SITE       "cdash.lanl.gov")
set( CTEST_DROP_LOCATION   "/submit.php?project=MCATK" )
set( CTEST_DROP_SITE_CDASH TRUE )
set( CTEST_CURL_OPTIONS CURLOPT_SSL_VERIFYPEER_OFF CURLOPT_SSL_VERIFYHOST_OFF )
