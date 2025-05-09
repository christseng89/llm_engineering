function quickSortUnique(arr) {
    if (arr.length <= 1) {
        return arr;
    }

    const pivotIndex = Math.floor(arr.length / 2);
    const pivot = arr[pivotIndex];
    const left = [];
    const right = [];

    for (let i = 0; i < arr.length; i++) {
        if (i === pivotIndex) continue;
        if (arr[i] < pivot) {
            left.push(arr[i]);
        } else if (arr[i] > pivot) {
            right.push(arr[i]);
        }
        // equal values are ignored (to skip duplicates)
    }

    return [...quickSortUnique(left), pivot, ...quickSortUnique(right)];
}

// Example usage:
const array = [3, 6, 8, 10, 1, 2, 1];
console.log(quickSortUnique(array)); // Output: [1, 2, 3, 6, 8, 10]
