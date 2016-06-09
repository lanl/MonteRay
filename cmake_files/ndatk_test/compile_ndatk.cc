#include "Exsdir.hh"

int main(int argc, char *argv[])
{
  ndatk::Exsdir x("exsdir");     // parse filename

//  // Print file header
//  std::cout << x.name() << std::endl;
//  for (int i = 0; i < x.number_of_events(); i++)
//    std::cout << x.event(i) << std::endl;
//
//  //Print first 10 entries
//
//  for (int i = 0; i != 10; i++) {
//    std::string id = x.table_identifier(i);
//    std::cout << id << ":"
//              << " " << x.file_name(id)
//              << " " << x.address(id)
//              << " " << x.atomic_weight(id)
//              << " " << x.temperature(id)
//              << std::endl;
//  }
  return 0;
}
