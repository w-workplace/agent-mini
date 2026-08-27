/**
 * quicksort.h - C++ 快速排序算法头文件
 * 
 * 提供多种快速排序实现：
 * 1. 基本快速排序（Lomuto分区）
 * 2. Hoare分区快速排序（更高效）
 * 3. 随机化快速排序（避免最坏情况）
 * 4. 三数取中快速排序（进一步优化）
 * 
 * 所有函数都是模板函数，支持任意可比较类型
 */

#ifndef QUICKSORT_H
#define QUICKSORT_H

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <type_traits>

namespace quicksort {

// ==================== 基本快速排序（Lomuto分区） ====================

/**
 * Lomuto分区函数
 * 选择最后一个元素作为基准
 */
template<typename T>
int partition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

/**
 * 快速排序递归实现
 */
template<typename T>
void quickSortRecursive(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSortRecursive(arr, low, pi - 1);
        quickSortRecursive(arr, pi + 1, high);
    }
}

/**
 * 快速排序主函数
 * 使用Lomuto分区方案
 */
template<typename T>
void quickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSortRecursive(arr, 0, arr.size() - 1);
    }
}

// ==================== Hoare分区快速排序 ====================

/**
 * Hoare分区函数
 * 通常比Lomuto分区更高效
 */
template<typename T>
int hoarePartition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[low];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        do {
            i++;
        } while (arr[i] < pivot);
        
        do {
            j--;
        } while (arr[j] > pivot);
        
        if (i >= j) {
            return j;
        }
        
        std::swap(arr[i], arr[j]);
    }
}

/**
 * Hoare分区快速排序递归实现
 */
template<typename T>
void quickSortHoareRecursive(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = hoarePartition(arr, low, high);
        quickSortHoareRecursive(arr, low, pi);
        quickSortHoareRecursive(arr, pi + 1, high);
    }
}

/**
 * Hoare分区快速排序主函数
 */
template<typename T>
void quickSortHoare(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSortHoareRecursive(arr, 0, arr.size() - 1);
    }
}

// ==================== 随机化快速排序 ====================

/**
 * 随机化分区函数
 * 随机选择基准，避免最坏情况
 */
template<typename T>
int randomizedPartition(std::vector<T>& arr, int low, int high) {
    int randomIndex = low + rand() % (high - low + 1);
    std::swap(arr[randomIndex], arr[high]);
    return partition(arr, low, high);
}

/**
 * 随机化快速排序递归实现
 */
template<typename T>
void randomizedQuickSortRecursive(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = randomizedPartition(arr, low, high);
        randomizedQuickSortRecursive(arr, low, pi - 1);
        randomizedQuickSortRecursive(arr, pi + 1, high);
    }
}

/**
 * 随机化快速排序主函数
 * 使用前需要调用srand(time(nullptr))初始化随机种子
 */
template<typename T>
void randomizedQuickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        randomizedQuickSortRecursive(arr, 0, arr.size() - 1);
    }
}

// ==================== 三数取中快速排序 ====================

/**
 * 选择三个元素的中位数作为基准
 */
template<typename T>
int medianOfThree(std::vector<T>& arr, int low, int high) {
    int mid = low + (high - low) / 2;
    
    // 对三个元素排序
    if (arr[high] < arr[low])
        std::swap(arr[low], arr[high]);
    if (arr[mid] < arr[low])
        std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[mid])
        std::swap(arr[mid], arr[high]);
    
    return mid; // 返回中位数的索引
}

/**
 * 三数取中分区函数
 */
template<typename T>
int medianPartition(std::vector<T>& arr, int low, int high) {
    int medianIndex = medianOfThree(arr, low, high);
    std::swap(arr[medianIndex], arr[high]);
    return partition(arr, low, high);
}

/**
 * 三数取中快速排序递归实现
 */
template<typename T>
void medianQuickSortRecursive(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = medianPartition(arr, low, high);
        medianQuickSortRecursive(arr, low, pi - 1);
        medianQuickSortRecursive(arr, pi + 1, high);
    }
}

/**
 * 三数取中快速排序主函数
 */
template<typename T>
void medianQuickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        medianQuickSortRecursive(arr, 0, arr.size() - 1);
    }
}

// ==================== 混合快速排序（小数组使用插入排序） ====================

/**
 * 插入排序（用于小数组）
 */
template<typename T>
void insertionSort(std::vector<T>& arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        T key = arr[i];
        int j = i - 1;
        
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/**
 * 混合快速排序递归实现
 * 当子数组大小小于阈值时使用插入排序
 */
template<typename T>
void hybridQuickSortRecursive(std::vector<T>& arr, int low, int high, int threshold = 16) {
    if (high - low + 1 <= threshold) {
        insertionSort(arr, low, high);
    } else if (low < high) {
        int pi = partition(arr, low, high);
        hybridQuickSortRecursive(arr, low, pi - 1, threshold);
        hybridQuickSortRecursive(arr, pi + 1, high, threshold);
    }
}

/**
 * 混合快速排序主函数
 * 结合了快速排序和插入排序的优点
 */
template<typename T>
void hybridQuickSort(std::vector<T>& arr, int threshold = 16) {
    if (!arr.empty()) {
        hybridQuickSortRecursive(arr, 0, arr.size() - 1, threshold);
    }
}

// ==================== 辅助函数 ====================

/**
 * 检查数组是否已排序
 */
template<typename T>
bool isSorted(const std::vector<T>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

/**
 * 生成随机整数数组
 */
template<typename T>
typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
generateRandomArray(size_t size, T minVal, T maxVal) {
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
    }
    
    std::vector<T> arr(size);
    for (size_t i = 0; i < size; i++) {
        arr[i] = minVal + rand() % (maxVal - minVal + 1);
    }
    return arr;
}

/**
 * 生成随机浮点数数组
 */
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
generateRandomArray(size_t size, T minVal, T maxVal) {
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
    }
    
    std::vector<T> arr(size);
    for (size_t i = 0; i < size; i++) {
        arr[i] = minVal + static_cast<T>(rand()) / RAND_MAX * (maxVal - minVal);
    }
    return arr;
}

} // namespace quicksort

#endif // QUICKSORT_H