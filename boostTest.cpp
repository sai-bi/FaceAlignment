/**
 * @author 
 * @version 2014/02/11
 */

#include <iostream>
#include <boost/regex.hpp>
#include <iostream>
#include <string>
using namespace std;

int main()
{
    std::string line;
    boost::regex pat( "^Subject: (Re: |Aw: )*(.*)" );

    while (std::cin)
    {
        std::getline(std::cin, line);
        boost::smatch matches;
        if (boost::regex_match(line, matches, pat))
            std::cout << matches[2] << std::endl;
    }
}




