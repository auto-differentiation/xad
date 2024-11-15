#include "../src/XAD/tools/Vector.hpp"
#include <gtest/gtest.h>


TEST(VectorPushBackTest, SingleElement) {
    ft::vector<int> vec;
    vec.push_back(42);
    ASSERT_EQ(vec.size(), 1);
    ASSERT_EQ(vec.capacity(), 1);
    ASSERT_EQ(vec[0], 42);
}

TEST(VectorPushBackTest, MultipleElements) {
    ft::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    ASSERT_EQ(vec.size(), 3);
    ASSERT_EQ(vec[0], 1);
    ASSERT_EQ(vec[1], 2);
    ASSERT_EQ(vec[2], 3);
}

TEST(VectorPushBackTest, CapacityDoubling) {
    ft::vector<int> vec;
    vec.push_back(1);
    size_t initial_capacity = vec.capacity();
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }
    ASSERT_GT(vec.capacity(), initial_capacity);
    ASSERT_EQ(vec.size(), 11);
}

TEST(VectorPushBackTest, ComplexType) {
    ft::vector<std::string> vec;
    vec.push_back("Hello");
    vec.push_back("World");
    ASSERT_EQ(vec.size(), 2);
    ASSERT_EQ(vec[0], "Hello");
    ASSERT_EQ(vec[1], "World");
}


TEST(VectorResizeTest, IncreaseSizeWithDefaultValue) {
    ft::vector<int> vec;
    vec.resize(5, 42);
    ASSERT_EQ(vec.size(), 5);
    for (size_t i = 0; i < vec.size(); ++i) {
        ASSERT_EQ(vec[i], 42);
    }
}

TEST(VectorResizeTest, IncreaseSizeWithoutValue) {
    ft::vector<int> vec;
    vec.resize(5);
    ASSERT_EQ(vec.size(), 5);
    for (size_t i = 0; i < vec.size(); ++i) {
        ASSERT_EQ(vec[i], 0);
    }
}

TEST(VectorResizeTest, DecreaseSize) {
    ft::vector<int> vec;
    vec.resize(5, 42);
    vec.resize(3);
    ASSERT_EQ(vec.size(), 3);
    ASSERT_EQ(vec[0], 42);
    ASSERT_EQ(vec[1], 42);
    ASSERT_EQ(vec[2], 42);
}

TEST(VectorResizeTest, ZeroSize) {
    ft::vector<int> vec;
    vec.resize(5, 42);
    vec.resize(0);
    ASSERT_EQ(vec.size(), 0);
    ASSERT_TRUE(vec.empty());
}

TEST(VectorResizeTest, LargerThanCapacity) {
    ft::vector<int> vec;
    vec.resize(5, 42);
    vec.resize(15, 99);
    ASSERT_EQ(vec.size(), 15);
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_EQ(vec[i], 42);
    }
    for (size_t i = 5; i < 15; ++i) {
        ASSERT_EQ(vec[i], 99);
    }
}


TEST(VectorClearTest, EmptyVector) {
    ft::vector<int> vec;
    vec.clear();
    ASSERT_EQ(vec.size(), 0);
    ASSERT_TRUE(vec.empty());
}

TEST(VectorClearTest, NonEmptyVector) {
    ft::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    ASSERT_EQ(vec.size(), 3);

    vec.clear();
    ASSERT_EQ(vec.size(), 0);
    ASSERT_TRUE(vec.empty());
}

TEST(VectorClearTest, PreserveCapacity) {
    ft::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    size_t capacity = vec.capacity();
    vec.clear();
    ASSERT_EQ(vec.capacity(), capacity);
}