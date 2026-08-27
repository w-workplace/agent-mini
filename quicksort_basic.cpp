#include <iostream>
#include <vector>
#include <algorithm>

/**
 * C++ 快速排序实现
 * 时间复杂度: 平均 O(n log n), 最坏 O(n²)
 * 空间复杂度: O(log n) 递归栈空间
 * 稳定性: 不稳定排序
 */

// 分区函数 - Lomuto分区方案
template<typename T>
int partition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[high];      // 选择最后一个元素作为基准
    int i = low - 1;          // 小于基准的元素的边界
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    std::swap(arr[i + 1], arr[high]);  // 将基准放到正确位置
    return i + 1;                      // 返回基准的最终位置
}

// 快速排序递归函数
template<typename T>
void quickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);  // 分区
        quickSort(arr, low, pivotIndex - 1);         // 递归排序左半部分
        quickSort(arr, pivotIndex + 1, high);        // 递归排序右半部分
    }
}

// 快速排序包装函数
template<typename T>
void quickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSort(arr, 0, arr.size() - 1);
    }
}

// 打印数组辅助函数
template<typename T>
void printArray(const std::vector<T>& arr) {
    for (const auto& elem : arr) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

// 主函数 - 演示使用
int main() {
    std::cout << "=== C++ 快速排序实现示例 ===\n" << std::endl;
    
    // 示例1: 整数数组排序
    std::vector<int> numbers = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "原始数组: ";
    printArray(numbers);
    
    quickSort(numbers);
    std::cout << "排序后数组: ";
    printArray(numbers);
    
    std::cout << std::endl;
    
    // 示例2: 浮点数数组排序
    std::vector<double> doubles = {3.14, 2.71, 1.41, 1.73, 0.0, -1.0};
    std::cout << "原始浮点数组: ";
    printArray(doubles);
    
    quickSort(doubles);
    std::cout << "排序后浮点数组: ";
    printArray(doubles);
    
    std::cout << std::endl;
    
    // 示例3: 字符串数组排序
    std::vector<std::string> words = {"banana", "apple", "cherry", "date", "fig"};
    std::cout << "原始字符串数组: ";
    printArray(words);
    
    quickSort(words);
    std::cout << "排序后字符串数组: ";
    printArray(words);
    
    std::cout << "\n=== 快速排序算法特点 ===" << std::endl;
    std::cout << "1. 分治算法: 将大问题分解为小问题解决" << std::endl;
    std::cout << "2. 原地排序: 不需要额外的存储空间" << std::endl;
    std::cout << "3. 不稳定排序: 相等元素的相对位置可能改变" << std::endl;
    std::cout << "4. 平均时间复杂度: O(n log n)" << std::endl;
    std::cout << "5. 最坏时间复杂度: O(n²) - 当数组已排序或逆序时" << std::endl;
    std::cout << "6. 优化方法: 随机化选择基准、三数取中法、小数组使用插入排序" << std::endl;
    
    return 0;
}