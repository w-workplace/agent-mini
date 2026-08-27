#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// 快速排序函数 - 使用Lomuto分区方案
template<typename T>
int partition(std::vector<T>& arr, int low, int high) {
    // 选择最后一个元素作为基准
    T pivot = arr[high];
    int i = low - 1; // 小于基准的元素的边界
    
    for (int j = low; j < high; j++) {
        // 如果当前元素小于或等于基准
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    // 将基准放到正确的位置
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// 快速排序递归函数
template<typename T>
void quickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        // 分区索引
        int pi = partition(arr, low, high);
        
        // 递归排序分区
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// 快速排序包装函数
template<typename T>
void quickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSort(arr, 0, arr.size() - 1);
    }
}

// 使用Hoare分区方案的快速排序（更高效）
template<typename T>
int hoarePartition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[low];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        // 从左边找到第一个大于等于基准的元素
        do {
            i++;
        } while (arr[i] < pivot);
        
        // 从右边找到第一个小于等于基准的元素
        do {
            j--;
        } while (arr[j] > pivot);
        
        // 如果指针相遇，返回分区点
        if (i >= j) {
            return j;
        }
        
        // 交换元素
        std::swap(arr[i], arr[j]);
    }
}

// 使用Hoare分区的快速排序
template<typename T>
void quickSortHoare(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = hoarePartition(arr, low, high);
        quickSortHoare(arr, low, pi);
        quickSortHoare(arr, pi + 1, high);
    }
}

template<typename T>
void quickSortHoare(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSortHoare(arr, 0, arr.size() - 1);
    }
}

// 随机化快速排序（避免最坏情况）
template<typename T>
int randomizedPartition(std::vector<T>& arr, int low, int high) {
    // 随机选择基准
    int randomIndex = low + rand() % (high - low + 1);
    std::swap(arr[randomIndex], arr[high]);
    
    return partition(arr, low, high);
}

template<typename T>
void randomizedQuickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = randomizedPartition(arr, low, high);
        randomizedQuickSort(arr, low, pi - 1);
        randomizedQuickSort(arr, pi + 1, high);
    }
}

template<typename T>
void randomizedQuickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        randomizedQuickSort(arr, 0, arr.size() - 1);
    }
}

// 打印数组
template<typename T>
void printArray(const std::vector<T>& arr) {
    for (const auto& elem : arr) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

// 测试函数
void testQuickSort() {
    std::cout << "=== 快速排序测试 ===\n" << std::endl;
    
    // 测试1: 整数数组
    {
        std::vector<int> arr = {10, 7, 8, 9, 1, 5};
        std::cout << "测试1 - 整数数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        // 验证排序
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ 排序正确" << std::endl;
        } else {
            std::cout << "✗ 排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试2: 浮点数数组
    {
        std::vector<double> arr = {3.14, 2.71, 1.41, 1.73, 0.0, -1.0};
        std::cout << "测试2 - 浮点数数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<double> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        std::vector<double> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ 排序正确" << std::endl;
        } else {
            std::cout << "✗ 排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试3: 随机大数组
    {
        const int SIZE = 1000;
        std::vector<int> arr(SIZE);
        std::srand(std::time(nullptr));
        
        for (int i = 0; i < SIZE; i++) {
            arr[i] = rand() % 10000;
        }
        
        std::cout << "测试3 - 随机大数组 (" << SIZE << "个元素):" << std::endl;
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        
        if (arrCopy == sorted) {
            std::cout << "✓ 大数组排序正确" << std::endl;
        } else {
            std::cout << "✗ 大数组排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试4: Hoare分区快速排序
    {
        std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
        std::cout << "测试4 - Hoare分区快速排序:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSortHoare(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ Hoare分区排序正确" << std::endl;
        } else {
            std::cout << "✗ Hoare分区排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试5: 随机化快速排序
    {
        std::vector<int> arr = {5, 2, 9, 1, 5, 6};
        std::cout << "测试5 - 随机化快速排序:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        randomizedQuickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ 随机化排序正确" << std::endl;
        } else {
            std::cout << "✗ 随机化排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试6: 已排序数组
    {
        std::vector<int> arr = {1, 2, 3, 4, 5, 6};
        std::cout << "测试6 - 已排序数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        if (arrCopy == arr) {
            std::cout << "✓ 已排序数组处理正确" << std::endl;
        } else {
            std::cout << "✗ 已排序数组处理错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试7: 逆序数组
    {
        std::vector<int> arr = {6, 5, 4, 3, 2, 1};
        std::cout << "测试7 - 逆序数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ 逆序数组排序正确" << std::endl;
        } else {
            std::cout << "✗ 逆序数组排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试8: 空数组
    {
        std::vector<int> arr;
        std::cout << "测试8 - 空数组:" << std::endl;
        std::cout << "数组大小: " << arr.size() << std::endl;
        
        quickSort(arr);
        std::cout << "排序后大小: " << arr.size() << std::endl;
        
        if (arr.empty()) {
            std::cout << "✓ 空数组处理正确" << std::endl;
        } else {
            std::cout << "✗ 空数组处理错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试9: 单元素数组
    {
        std::vector<int> arr = {42};
        std::cout << "测试9 - 单元素数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        if (arrCopy == arr) {
            std::cout << "✓ 单元素数组处理正确" << std::endl;
        } else {
            std::cout << "✗ 单元素数组处理错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 测试10: 重复元素数组
    {
        std::vector<int> arr = {5, 2, 5, 1, 5, 5, 2};
        std::cout << "测试10 - 重复元素数组:" << std::endl;
        std::cout << "原始数组: ";
        printArray(arr);
        
        std::vector<int> arrCopy = arr;
        quickSort(arrCopy);
        std::cout << "排序后: ";
        printArray(arrCopy);
        
        std::vector<int> sorted = arr;
        std::sort(sorted.begin(), sorted.end());
        if (arrCopy == sorted) {
            std::cout << "✓ 重复元素数组排序正确" << std::endl;
        } else {
            std::cout << "✗ 重复元素数组排序错误" << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "=== 所有测试完成 ===" << std::endl;
}

int main() {
    // 初始化随机种子
    std::srand(std::time(nullptr));
    
    // 运行测试
    testQuickSort();
    
    return 0;
}