#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include "iterators/vector_Iterator.hpp"
#include "iterators/reverse_iterator.hpp"
#include "enable_if.hpp"
#include "compare.hpp"

namespace ft {

template < class T, class Alloc = std::allocator<T> > 
class vector
{
    public :

    typedef T value_type;
    typedef Alloc allocator_type;
    typedef typename allocator_type::reference reference;
    typedef typename allocator_type::const_reference const_reference;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef ft::Ra_iterator<pointer> iterator;
    typedef ft::Ra_iterator<const_pointer> const_iterator;
    typedef ft::reverse_iterator<iterator> reverse_iterator;
    typedef ft::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef typename ft::iterator_traits<iterator>::difference_type	difference_type;
    typedef std::size_t     size_type;

    private :
        pointer _arr;
        allocator_type alloc;
        size_type       _current;
        size_type       _capacity;
    
    public :

     /** ************************************************************************** */
	 /**     MEMBER FUNCTION  : CONSTRUCTORS & DESTRUCTOR                           */
	 /** ************************************************************************** */

   explicit vector (const allocator_type& alloc1 = allocator_type())
   {
            _capacity = 0;
            _current = 0;
            _arr = nullptr;
            this->alloc = alloc1;
   }
	
    explicit vector (size_type n, const value_type& val = value_type(),
                 const allocator_type& alloc1 = allocator_type())
                 {
                    _arr = this->alloc.allocate(n);
                    _current = n;
                    _capacity = n;
                    this->alloc = alloc1;
                    for (size_type i = 0; i < n; i++)
                        this->alloc.construct(_arr + i , val);
                       // std::cout << "adasd" << std::endl;
                 }
	
    template <class Iterator>
    void vector_impl(Iterator first, Iterator last,std::input_iterator_tag)
    {
         _current = 0;
        _capacity = 0;
        while (first != last)
        {
           push_back(first);
           first++;
        }
    }

     template <class Iterator>
    void vector_impl(Iterator first, Iterator last, std::random_access_iterator_tag)
    {
        Iterator f = first;
        _current = 0;
        _capacity = 0;
        while (f != last)
        {
            _current++;
            _capacity++;
            f++;
        }
        _arr = this->alloc.allocate(_capacity);
        int i = 0;
        while (first != last)
        {
            this->alloc.construct((_arr + i), *first);
            first++;
            i++;
        }
    }
    
    template <class InputIterator>
    vector (InputIterator first, typename ft::enable_if<!is_integral<InputIterator>::value, InputIterator >::type last,
                 const allocator_type& alloc1 = allocator_type()) 
    {
        this->alloc = alloc1;
                    // this->vector_impl(first, last, typename iterator_traits<InputIterator>::iterator_category());
        InputIterator f = first;
        _current = 0;
        _capacity = 0;
        while (f != last)
        {
            _current++;
            _capacity++;
            f++;
        }
        _arr = this->alloc.allocate(_capacity);
        int i = 0;
        while (first != last)
        {
            this->alloc.construct((_arr + i), *first);
            first++;
            i++;
        }
    }
	
    // Move Constructor
vector(vector&& x) noexcept
{
    this->_arr = x._arr;
    this->_current = x._current;
    this->_capacity = x._capacity;
    this->alloc = std::move(x.alloc);

    // Reset the source object
    x._arr = nullptr;
    x._current = 0;
    x._capacity = 0;
}

// Move Constructor with Allocator
vector(vector&& x, const allocator_type& alloc1) : alloc(alloc1)
{
    if (alloc == x.alloc)
    {
        // If the allocators are equal, we can take ownership
        this->_arr = x._arr;
        this->_current = x._current;
        this->_capacity = x._capacity;

        // Reset the source object
        x._arr = nullptr;
        x._current = 0;
        x._capacity = 0;
    }
    else
    {
        // If the allocators differ, allocate and copy
        this->_arr = this->alloc.allocate(x._current);
        this->_current = x._current;
        this->_capacity = x._current;
        for (size_type i = 0; i < x._current; ++i)
        {
            this->alloc.construct(this->_arr + i, std::move(x._arr[i]));
            x.alloc.destroy(x._arr + i);
        }
        x.alloc.deallocate(x._arr, x._capacity);
        x._arr = nullptr;
        x._current = 0;
        x._capacity = 0;
    }
}
    vector (const vector& x)
    {
        this->_arr = nullptr;
		this->_current = 0;
		this->_capacity = 0;
		this->alloc = allocator_type();
        *this = x;
    }

    ~vector() {
        if (_arr != nullptr) {
        for (size_type i = 0; i < _current; i++)
           alloc.destroy(_arr + i);
        alloc.deallocate(_arr, _capacity);
        }
    }
    vector& operator=(vector&& x) noexcept
{
    if (this != &x) // Check for self-assignment
    {
        // Free existing resources
        if (_arr != nullptr)
        {
            for (size_type i = 0; i < _current; ++i)
                alloc.destroy(_arr + i);
            alloc.deallocate(_arr, _capacity);
        }

        // Transfer ownership of resources
        _arr = x._arr;
        _current = x._current;
        _capacity = x._capacity;
        alloc = std::move(x.alloc);

        // Reset the source object
        x._arr = nullptr;
        x._current = 0;
        x._capacity = 0;
    }
    return *this;
}

    
    vector& operator=(const vector& c)
    {
        if (*this != c)
        {
            for (size_t i = 0; i < _current; i++)
                this->alloc.destroy(_arr + i);
            if (_capacity > 0)
                this->alloc.deallocate(_arr, _capacity);
           
            if (this->_capacity < c._capacity)
                this->_capacity = c._current;
            //this->alloc = c.alloc;
            this->_current = c._current;
            _arr = this->alloc.allocate(_capacity);
            for (size_t i = 0; i < c._current;i++)
                this->alloc.construct(_arr + i, c._arr[i]);
        }
        return *this;
    }

     /** ************************************************************************** */
	 /**                               ITERATORS                                    */
	 /** ************************************************************************** */

    iterator begin()
    {
        return iterator(_arr);
    }

    const_iterator begin() const
    {
        return (const_iterator(_arr));
    }

    iterator end()
    {
        return (iterator(_arr + _current));
    }
    const_iterator end() const
    {
        return (const_iterator(_arr + _current));
    }
    reverse_iterator rbegin()
    {
        return (reverse_iterator(_arr + _current));
    }
    const_reverse_iterator rbegin() const
    {
        return (const_reverse_iterator(_arr + _current));
    }
    reverse_iterator rend()
    {
        return (reverse_iterator(begin()));
    }
    const_reverse_iterator rend() const
    {
        return (const_reverse_iterator(begin()));
    }
     /** ************************************************************************** */
	 /**                               CAPACITY                                     */
	 /** ************************************************************************** */

    size_type size() const 
    {
        return _current;
    }

    size_type max_size() const
    {
        return alloc.max_size();
    }

    void resize (size_type n, value_type val = value_type()) /// resize the the vector (destroy extra element if n < current size) else reallocate and insert val;
    {
        if (n > max_size())
            throw std::length_error("vector::resize");
        else if (n < _current)
        {
            // for (size_type i = n; i < _current;i++)
            //     this->alloc.destroy(_arr + i);
            //this->alloc.deallocate(_arr + n, _capacity);
            while (n != size())
                pop_back();
           
        }
        else
        {
            // if (n > _capacity * 2)
                // reserve(n);
            // else
            //     reserve(_capacity * 2);
            // while (n  != size()) 
            //     push_back(val);
            if (n > _capacity * 2)
                reserve(n);
            else if (n > _capacity)
                reserve(_capacity * 2);
            while (n != size())
                insert(this->end(), n - size(), val);
            //reserve(n);
        }
       // _current = n;
    }

    size_type capacity() const
    {
        return _capacity;
    }
    bool empty() const
    {
        if (_current == 0)
            return (true);
        return false;
    }
    void reserve (size_type n)              // reallocate the array with n 
    {
         if (n > max_size())
            throw std::length_error("vector::resize");
        if (n > _capacity)
        {
            pointer tmp = nullptr;
            tmp = this->alloc.allocate(n);
            size_type i;
            for (i = 0; i < this->size(); i++)
            {
                this->alloc.construct(tmp + i, _arr[i]);
                this->alloc.destroy(_arr + i);
            }
           // std::cout << "i = " <<  i << std::endl;
            this->alloc.deallocate(_arr, _capacity);
            _arr = nullptr;
            _arr = tmp;
            _capacity = n;
           
        }

    }


    void shrink_to_fit()            //// change the allocation size to the _current aka size of vector
    {
        if (_capacity != _current)
        {
            pointer tmp = nullptr;
            tmp = this->alloc.allocate(_current);
            size_type i;
            for (i = 0; i < this->size(); i++)
            {

                this->alloc.construct(tmp + i, _arr[i]);
                this->alloc.destroy(_arr + i);
            }
            this->alloc.deallocate(_arr, _capacity);
            _arr = tmp;
            _capacity = _current;
        }
    }

    /** ************************************************************************** **/
	/**                               ELEMENT ACCESS                               **/
	/** ************************************************************************** **/
       reference operator[] (size_type n)
       {
           return  _arr[n];
       }

        const_reference operator[] (size_type n) const
        {
            return _arr[n];
        }

        reference at (size_type n)
        {
            std::string str = "vector";
            if (n >= size())
                    throw std::out_of_range(str);
            return _arr[n];
        }
        const_reference at (size_type n) const
        {
            std::string str = "vector";
            if (n >= size())
                    throw std::out_of_range(str);
            return _arr[n];
        }
        reference front()
        {
            return (_arr[0]);

        }
        const_reference front() const
        {
            return (_arr[0]);
        }
        reference back()
        {
            return (_arr[size() - 1]);
        }
        const_reference back() const
        {
            return (_arr[size() - 1]);
        }

    /** ************************************************************************** **/
	/**                               Modifiers                                    **/
	/** ************************************************************************** **/

    template <class InputIterator>
    void assign (InputIterator first, typename ft::enable_if<!is_integral<InputIterator>::value, InputIterator >::type last)
    {
        InputIterator f = first;
        size_type n = 0;

        while (f != last)
        {
            f++;
            n++;
        }
        if (n != 0)
        {
            if (n > _capacity)
            {
                reserve(n);
                _capacity = n;
            }
            for (size_type i = 0; i < n; i++)
            {
                if (i < size())
                    this->alloc.destroy(_arr + i);
                this->alloc.construct(_arr + i, *first);
                first++;
            }
            _current = n;
        }
    }
	
    void assign (size_type n, const value_type& val)
    {
        if (n > _capacity)
            reserve(n);
        for (size_type i = 0; i < n; i++)
        {
            if (i < size())
                this->alloc.destroy(_arr + i);
            this->alloc.construct(_arr + i, val);
        }
        _current = n;
    }

    void push_back (const value_type& val)
    {
        if (empty())
            reserve(1);
        else if (_current + 1 > _capacity)
            reserve(_capacity * 2);
        this->alloc.construct(_arr + _current, val);
        _current++;
    }

    void pop_back()
    {
        this->alloc.destroy(_arr + _current);
        _current--;
    }

    iterator insert (iterator position, const value_type& val)
    {
        iterator a = this->begin();
        size_type n = 0;
        if (empty())
            reserve(1);
        else if (_current + 1 > _capacity)
            reserve(_capacity * 2);
        while (a != position)
        {
            n++;
            a++;
        }
        for (size_t i = size(); i > n; --i)
            this->alloc.construct(_arr + i, _arr[i - 1]);
        this->alloc.construct(_arr + n, val);
        _current++;
        return (iterator(_arr + n));
    }

    void insert (iterator position, size_type n, const value_type& val)
    { 
        iterator a = begin();
        difference_type e = std::distance(a, position);
        if (empty())
            reserve(n);
        else if (n + size() > _capacity)
        {
            if (n > size())
                reserve(n + size());
            else
                reserve(_capacity * 2);    
        }
        for (difference_type i = size() - 1; i >= e; --i)
            this->alloc.construct(_arr + (i + n), _arr[i]);
        for (size_type i = 0; i < n ;++i)
            this->alloc.construct(_arr + e++, val);
        _current+=n;
    }
    
    template <class InputIterator>
    void insert (iterator position, InputIterator first, typename ft::enable_if<!is_integral<InputIterator>::value, InputIterator >::type last)
    {
        InputIterator a = first;
        difference_type e = std::distance(this->begin(),position);
        difference_type n =std::distance(a, last);
        if (empty())
            reserve(n);
        else if ((size_type)(n + size()) > _capacity)
        {
            if (n + size() > _capacity * 2)
                reserve(n + size());
            else
                reserve(_capacity * 2);
        }
        for (difference_type i = size() - 1; i >= e; --i)
            this->alloc.construct(_arr + (i + n), _arr[i]);
        for (size_t i = 0; i < (size_t)n ;++i)
        {
            this->alloc.construct(_arr + e++, *first);
            first++;
        }
        _current+=n;
    }

    iterator erase (iterator position)
    {
        size_type n = std::distance(this->begin(), position);
        this->alloc.destroy(_arr + n);
        for (size_type i = n; i < size() - 1;i++)
            this->alloc.construct(_arr + i, _arr[i + 1]);
        _current--;
        return(iterator(_arr + n));
    }

    iterator erase (iterator first, iterator last)
    {
        // difference_type			index = std::distance(begin(), first);
        // for (; first != last - 1; ++first)
        //     this->erase(first);
        // return (iterator(_arr + index));
        difference_type		range = std::distance(first, last);
		difference_type		index = std::distance(begin(), first);
		for (difference_type i = index; i < range;++i)
			this->alloc.destroy(this->_arr + i);
		this->_current -= range;
		for (size_type i = index;i < this->_current;++i, ++range)
			this->alloc.construct(this->_arr + i, this->_arr[range]);
		return (iterator(this->_arr + index));

    }
    // void swap (vector& x)
    // {
    //     vector temp;

    //     temp.assign(this->begin(), this->end());
    //     this->assign(x.begin(), x.end());
    //     x.assign(temp.begin(), temp.end());
    // }

    void swap (vector& x)
	{
		pointer tmp = this->_arr;
		size_type _size = this->size();
		size_type _capacity1 = this->_capacity;
		this->_arr = x._arr;
		this->_current = x._current;
		this->_capacity = x._capacity;
		x._arr =  tmp;
		x._current = _size;
		x._capacity = _capacity1;
	}

    void clear()
    {
        erase(this->begin(), this->end());
    }

    allocator_type get_allocator() const
    {
        return (this->alloc);
    }


    }; /// end class vector

        template <class T, class Alloc>
        bool operator== (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
            if (lhs.size() == rhs.size())
                return (std::equal(lhs.begin(),lhs.end(),rhs.begin()));
            else
                return (false);
        }

        template <class T, class Alloc>
        bool operator!= (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
            return (!(lhs == rhs));
        }
        template <class T, class Alloc>
        bool operator<  (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
                return (std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end()));
        }
        template <class T, class Alloc>
        bool operator<= (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
            return ( !(rhs < lhs));
        }
        template <class T, class Alloc>
        bool operator>  (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
            return (rhs < lhs);
        }
        template <class T, class Alloc>
        bool operator>= (const vector<T,Alloc>& lhs, const vector<T,Alloc>& rhs)
        {
            return (!(lhs < rhs));
        }
        template <class T, class Alloc>
        void swap (vector<T,Alloc>& x, vector<T,Alloc>& y)
        {
        	x.swap(y);
        }
    } //end namespace
#endif