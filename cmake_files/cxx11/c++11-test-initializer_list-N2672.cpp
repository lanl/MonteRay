#include <list>
#include <complex>
#include <vector>
#include <string>
#include <map>
#include <initializer_list>
#include <iostream>

std::list< std::string > func( void ) { return { "Norah", "Jane" }; }

template <class T>
struct S {
    std::vector<T> v;
    S( void ) {}
    S(std::initializer_list<T> l) : v(l) {
         std::cout << "constructed with a " << l.size() << "-element list\n";
    }
    void append(std::initializer_list<T> l) {
        v.insert(v.end(), l.begin(), l.end());
    }
    std::pair<const T*, std::size_t> c_arr() const {
        return {&v[0], v.size()};  // list-initialization in return statement
                                   // this is NOT a use of std::initializer_list
    }
};

template <typename T>
void templated_fn(T) {}

class A {
public:
    A( void ) {}
    A( int ) {}
};
void
listWithAuto()
{
    S<int> s = {1, 2, 3, 4, 5}; // direct list-initialization
    s.append({6, 7, 8});      // list-initialization in function call

    std::cout << "The vector size is now " << s.c_arr().second << " ints:\n";

#if HAS_CXX11_AUTO
    for (auto n : s.v) std::cout << ' ' << n;

    std::cout << '\n';

    std::cout << "range-for over brace-init-list: \n";

    for (int x : {-1, -2, -3}) // the rule for auto makes this ranged for work
        std::cout << x << ' ';
    std::cout << '\n';

    auto al = {10, 11, 12};   // special rule for auto

    std::cout << "The list bound to auto has size() = " << al.size() << '\n';
#endif
//    templated_fn({1, 2, 3}); // compiler error! "{1, 2, 3}" is not an expression,
                             // it has no type, and so T cannot be deduced
    templated_fn<std::initializer_list<int>>({1, 2, 3}); // OK
    templated_fn<std::vector<int>>({1, 2, 3});           // also OK

}

int main()
{
    int intInit = {1};

    std::complex<double> z{1,2};

    std::vector<std::string>* pv = new std::vector<std::string>{"once", "upon", "a", "time"}; // 4 string elements
    delete pv;

    std::list<std::string> l = func(); // return list

    int* e {}; // initialization to zero / null pointer

    double x = double{1}; // explicitly construct a double

    std::map<std::string,int> anim = { {"bear",4}, {"cassowary",2}, {"tiger",7} };

    listWithAuto();

    // A a = A();  // shouldn't compile.  thinks this is a function declaration
    A a = A{};

    return 0;
}
