#include <memory>

int main( void ) {

    std::unique_ptr<int> p( new int );
    *p = 99;

    std::unique_ptr<int> p2;
    p2 = std::move( p );

    int test = *p2 == 99 ? 0 : 1;
    return test;
}
