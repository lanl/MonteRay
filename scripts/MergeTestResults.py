#!/bin/env python
#
# Merges CTest xml files together that contain testing results
#
#
import xml.etree.ElementTree as ET

import glob, os

def getTestMeta( tree ): # Gets the relevant meta information from the xml file
    """Return a list of meta information"""
    root    = tree.getroot()
    Testing = root.find( "Testing" )
    result = {}
    result[ "Testing"   ] = Testing
    result[ "TestList"  ] = Testing.find( "TestList" )
    result[ "StartTime" ] = int( Testing.find( "StartTestTime" ).text )
    result[ "StartDate" ] = Testing.find( "EndDateTime" ).text
    result[ "EndTime"   ] = int( Testing.find( "EndTestTime" ).text )
    result[ "EndDate"   ] = Testing.find( "EndDateTime" ).text
    result[ "Elapsed"   ] = float( Testing.find( "ElapsedMinutes" ).text )
    return result

errors = []
MergedName = "Test.xml"

# Deleting any previous merge if it exists
try:
    os.remove( MergedName )
except (IOError, os.error), why:
    errors.append(str(why))

TestSummaries = glob.glob( "Test*.xml" )

mergedXML = ET.parse( TestSummaries.pop() )
base = getTestMeta( mergedXML )
MergedList  = base[ "TestList" ]
MergedTests = base[ "Testing"  ]
StartTime   = base[ "StartTime" ]
EndTime     = base[ "EndTime" ]
TotalElapsed = base[ "Elapsed" ]

for TS in TestSummaries:
    NextXML = ET.parse( TS )
    next = getTestMeta( NextXML )

    # Merge the test names
    TestList = next[ "TestList" ]
    for t in TestList.findall( "Test" ):
        MergedList.append( t )
  
    # Merge the test results
    Testing  = next[ "Testing" ];
    for t in Testing.findall( "Test" ):
        MergedTests.append( t )

    if next[ "EndTime" ] > EndTime:
        EndTime = next[ "EndTime" ]
        MergedTests.find( "EndTestTime" ).text = str( EndTime )
        MergedTests.find( "EndDateTime" ).text = next[ "EndDate" ]

    if next[ "StartTime" ] < StartTime:
        StartTime = next[ "StartTime" ]
        MergedTests.find( "StartTestTime" ).text = str( StartTime )
        MergedTests.find( "StartDateTime" ).text = next[ "StartDate" ]

    TotalElapsed += next[ "Elapsed" ]

    MergedTests.find( 'ElapsedMinutes' ).text = str( TotalElapsed )

# translate non-utf characters (thank you python!)
for t in MergedTests.findall( "Test" ):
    value = t.find( "Results" ).find("Measurement").find( "Value" )
    if not value.text:
        continue
    pos = value.text.find( '[NON-UTF-8-BYTE-0x' )
    if pos == -1:
        continue
    print 'Fixing ',t.find( "Name" ).text

    value.text = value.text.replace( '[NON-UTF-8-BYTE-0xB0]', u'\xB0' )
    value.text = value.text.replace( '[NON-UTF-8-BYTE-0xB1]', u'\xB1' )

mergedXML.write( MergedName )

