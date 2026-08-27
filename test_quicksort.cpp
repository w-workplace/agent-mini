#include "quicksort.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// 打印数组
template<typename T>
void printArray(const std::vector<T>& arr, const std::string& label = "") {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    
    if (arr.size() <= 20) {
        for (const auto& elem : arr) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[前10个元素] ";
        for (size_t i = 0; i < 10 && i < arr.size(); i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << "... [共" << arr.size() << "个元素]" << std::endl;
    }
}

// 测试排序算法性能（基本版本）
template<typename T>
void testSortPerformance(std::vector<T> arr, const std::string& algorithmName, 
                         void (*sortFunc)(std::vector<T>&)) {
    auto start = std::chrono::high_resolution_clock::now();
    sortFunc(arr);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool sorted = quicksort::isSorted(arr);
    std::cout << algorithmName << ": " 
              << duration.count() << " 微秒, "
              << (sorted ? "✓ 排序正确" : "✗ 排序错误") << std::endl;
}

// 测试排序算法性能（带阈值参数的版本）
template<typename T>
void testSortPerformanceWithThreshold(std::vector<T> arr, const std::string& algorithmName, 
                                      void (*sortFunc)(std::vector<T>&, int), int threshold = 16) {
    auto start = std::chrono::high_resolution_clock::now();
    sortFunc(arr, threshold);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool sorted = quicksort::isSorted(arr);
    std::cout << algorithmName << ": " 
              << duration.count() << " 微秒, "
              << (sorted ? "✓ 排序正确" : "✗ 排序错误") << std::endl;
}

int main() {
    // 初始化随机种子
    std::srand(std::time(nullptr));
    
    std::cout << "=== C++ 快速排序算法库测试 ===\n" << std::endl;
    
    // 测试1: 基本功能测试
    {
        std::cout << "1. 基本功能测试:" << std::endl;
        
        std::vector<int> arr1 = {64, 34, 25, 12, 22, 11, 90};
        std::vector<int> arr2 = arr1;
        std::vector<int> arr3 = arr1;
        std::vector<int> arr4 = arr1;
        std::vector<int> arr5 = arr1;
        
        printArray(arr1, "原始数组");
        
        quicksort::quickSort(arr1);
        printArray(arr1, "基本快速排序");
        
        quicksort::quickSortHoare(arr2);
        printArray(arr2, "Hoare快速排序");
        
        quicksort::randomizedQuickSort(arr3);
        printArray(arr3, "随机化快速排序");
        
        quicksort::medianQuickSort(arr4);
        printArray(arr4, "三数取中快速排序");
        
        quicksort::hybridQuickSort(arr5);
        printArray(arr5, "混合快速排序");
        
        std::cout << std::endl;
    }
    
    // 测试2: 不同类型数据测试
    {
        std::cout << "2. 不同类型数据测试:" << std::endl;
        
        std::vector<double> doubles = {3.14, 2.71, 1.41, 1.73, 0.0, -1.0};
        printArray(doubles, "原始浮点数组");
        quicksort::quickSort(doubles);
        printArray(doubles, "排序后浮点数组");
        
        std::vector<std::string> words = {"banana", "apple", "cherry", "date", "fig"};
        printArray(words, "原始字符串数组");
        quicksort::quickSort(words);
        printArray(words, "排序后字符串数组");
        
        std::cout << std::endl;
    }
    
    // 测试3: 边界情况测试
    {
        std::cout << "3. 边界情况测试:" << std::endl;
        
        std::vector<int> emptyArr;
        std::vector<int> singleArr = {42};
        std::vector<int> sortedArr = {1, 2, 3, 4, 5};
        std::vector<int> reverseArr = {5, 4, 3, 2, 1};
        std::vector<int> duplicateArr = {5, 2, 5, 1, 5, 5, 2};
        
        quicksort::quickSort(emptyArr);
        std::cout << "空数组大小: " << emptyArr.size() << std::endl;
        
        quicksort::quickSort(singleArr);
        printArray(singleArr, "单元素数组");
        
        quicksort::quickSort(sortedArr);
        printArray(sortedArr, "已排序数组");
        
        quicksort::quickSort(reverseArr);
        printArray(reverseArr, "逆序数组");
        
        quicksort::quickSort(duplicateArr);
        printArray(duplicateArr, "重复元素数组");
        
        std::cout << std::endl;
    }
    
    // 测试4: 性能测试
    {
        std::cout << "4. 性能测试 (10000个随机整数):" << std::endl;
        
        std::vector<int> randomArr = quicksort::generateRandomArray<int>(10000, 0, 10000);
        
        testSortPerformance(randomArr, "基本快速排序", quicksort::quickSort);
        testSortPerformance(randomArr, "Hoare快速排序", quicksort::quickSortHoare);
        testSortPerformance(randomArr, "随机化快速排序", quicksort::randomizedQuickSort);
        testSortPerformance(randomArr, "三数取中快速排序", quicksort::medianQuickSort);
        testSortPerformanceWithThreshold(randomArr, "混合快速排序", quicksort::hybridQuickSort);
        
        std::cout << std::endl;
    }
    
    // 测试5: 最坏情况测试（已排序大数组）
    {
        std::cout << "5. 最坏情况测试 (10000个已排序整数):" << std::endl;
        
        std::vector<int> sortedLarge(10000);
        for (int i = 0; i < 10000; i++) {
            sortedLarge[i] = i;
        }
        
        // 注意：基本快速排序在已排序数组上性能最差
        // 随机化和三数取中版本应该表现更好
        testSortPerformance(sortedLarge, "基本快速排序", quicksort::quickSort);
        testSortPerformance(sortedLarge, "随机化快速排序", quicksort::randomizedQuickSort);
        testSortPerformance(sortedLarge, "三数取中快速排序", quicksort::medianQuickSort);
        
        std::cout << std::endl;
    }
    
    // 测试6: 小数组优化测试
    {
        std::cout << "6. 小数组优化测试:" << std::endl;
        
        std::vector<int> smallArr = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        printArray(smallArr, "原始小数组");
        
        // 使用较小的阈值测试混合排序
        quicksort::hybridQuickSort(smallArr, 5);
        printArray(smallArr, "混合排序(阈值=5)");
        
        std::cout << std::endl;
    }
    
    std::cout << "=== 测试完成 ===" << std::endl;
    std::cout << "\n快速排序算法总结:" << std::endl;
    std::cout << "1. 基本快速排序: 简单易懂，适合教学" << std::endl;
    std::cout << "2. Hoare快速排序: 交换次数更少，效率更高" << std::endl;
    std::cout << "3. 随机化快速排序: 避免最坏情况，性能稳定" << std::endl;
    std::cout << "4. 三数取中快速排序: 进一步优化基准选择" << std::endl;
    std::cout << "5. 混合快速排序: 结合插入排序，对小数组更优" << std::endl;
    
    return 0;
}