#include <iostream>
#include <cstdlib>

#include "framework.h"

/*
 * Each one of these contain a model and a run() function to run the framework for that model.
 * Obviously include only one.
*/
// #include "mogi1.h"
// #include "mogi2.h"
#include "okada1.h"
// #include "okada2.h"
// #include "okada3.h"
// #include "test.h"

#define RESULTS_THRESHOLD 1e-13

using namespace std;

int main(int argc, char** argv){

    // Function in mogi*.h or okada*.h included above
    run(argc, argv);

    // fflush(stdout);
    // if (!parameters.benchmark) {
    //     // Test the outputs
    //
    //     DATA_TYPE point[2];
    //     DATA_TYPE step[2] = {
    //         abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
    //         abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
    //     };
    //     MyModel model = MyModel();
    //
    //     int correctCount = 0;
    //     bool shouldBeInList, isInList;
    //     list = framework.getList(&length);
    //
    //     printf("\nVerifying results...\n");
    //
    //     for (unsigned int i = 0; i < limits[0].N; i++) {
    //         point[0] = limits[0].lowerLimit + i * step[0];
    //
    //         for (unsigned int j = 0; j < limits[1].N; j++) {
    //             point[1] = limits[1].lowerLimit + j * step[1];
    //
    //             // Check if it SHOULD be in list
    //             shouldBeInList = model.toBool(model.validate_cpu(point, parameters.dataPtr));
    //             if(shouldBeInList)
    //                 correctCount++;
    //
    //             // Check the list to see if the point is there
    //             isInList = false;
    //             for(int k=0; k<length; k++){
    //                 if(list[k*2] == point[0] && list[k*2 + 1] == point[1]){
    //                     isInList = true;
    //                     break;
    //                 }
    //             }
    //
    //             if(shouldBeInList ^ isInList)
    //                 printf("(%f, %f): shouldBeInList=%s, isInList=%s\n", point[0], point[1], shouldBeInList ? "true" : "false", isInList ? "true" : "false");
    //         }
    //     }
    //
    //     if(correctCount != length){
    //         printf("List length is %d but should be \n", length, correctCount);
    //
    //         printf("Points is list:\n");
    //         for(int k=0; k<length; k++){
    //             printf("%f, %f\n", list[2*k], list[2*k + 1]);
    //         }
    //     }
    // }

    return 0;
}
