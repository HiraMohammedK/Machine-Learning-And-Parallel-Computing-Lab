#include <iostream>
#include <thread>
using namespace std;

void printNumbers(int n) {
    for (int i = 1; i <= n; ++i) {
        cout << i << " ";
    }
    cout << endl;
}

int main() {
    int n;

    cout << "Enter the value of n: ";
    cin >> n;

    thread numThread(printNumbers, n);

    numThread.join();

    return 0;
}
