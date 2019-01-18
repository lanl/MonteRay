#The commandline [bash gitMONTERAY.bash args....]

GIT_EXECUTABLE=${1}

if [ ${2} != UPDATE ]; then
  # This will be called by ctest_start (set CTEST_CHECKOUT_COMMAND), 
  # so it does not matter the location executed. 

  BranchRequest="--branch ${2}"
  if [ ${2} = NOBRANCH ]; then
     BranchRequest=""
  fi
  repo=${3}
  CTEST_SOURCE_DIRECTORY=${4}

  ${GIT_EXECUTABLE} clone --recursive ${BranchRequest} ${repo} ${CTEST_SOURCE_DIRECTORY}
  ${GIT_EXECUTABLE} -C ${CTEST_SOURCE_DIRECTORY} submodule foreach --recursive "git checkout master"

elif [ ${2} = UPDATE ]; then
  # This will be called by ctest_update (set CTEST_GIT_UPDATE_CUSTOM). 
  # It sees its execution location local to .git/
  
  ${GIT_EXECUTABLE} pull --quiet
  ${GIT_EXECUTABLE} submodule foreach --recursive "git pull"

fi

exit
