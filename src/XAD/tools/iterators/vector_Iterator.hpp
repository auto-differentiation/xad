#ifndef RANDOM_ACCESS_ITERATOR_HPP
# define RANDOM_ACCESS_ITERATOR_HPP

//#include "utils.hpp"
#include "iterator_traits.hpp"

namespace ft
{
    // RANDOM_ACCESS_ITERATOR class
    template <class T>
    class Ra_iterator
    {
        public :   
            typedef T         iterator_type;
		    typedef typename ft::iterator_traits<T>::value_type			value_type;
		    typedef typename ft::iterator_traits<T>::pointer			pointer;
		    typedef typename ft::iterator_traits<T>::reference			reference;
		    typedef typename ft::iterator_traits<T>::difference_type	difference_type;
		    typedef typename ft::iterator_traits<T>::iterator_category iterator_category;
        private :
            iterator_type it;

        public :
           	/** ************************************************************************** */
		/**                                CONSTRUCTORS  AND DESTRUCTORS               */
		/** ************************************************************************** */

                Ra_iterator() {it = NULL;}
                Ra_iterator(iterator_type a) : it(a) {}
                template <typename it1>
                Ra_iterator(const Ra_iterator<it1>& a) : it(a.base()) {} 
                ~Ra_iterator(){}
           ////////////////////////////////////////////////////



        /** ************************************************************************** */
		/**                                  MEMBERS                                   */
		/** ************************************************************************** */




            iterator_type base() const { return it; }

        /** ************************************************************************** */
		/**                                COMPARAISON OPERATORS PROTOTYPES            */
		/** ************************************************************************** */
        Ra_iterator &operator=(Ra_iterator const &c)
        {
            it = c.it;
            return *this;
        }

        template <typename it1, typename it2>
        friend bool operator==(const Ra_iterator<it1> &, const Ra_iterator <it2> &);

        template <typename it1, typename it2>
        friend bool operator!=(const Ra_iterator<it1> &, const Ra_iterator <it2> &);

        template <typename it1, typename it2>
        friend bool operator>=(const Ra_iterator<it1> &, const Ra_iterator <it2> &);

        template <typename it1, typename it2>
        friend bool operator<=(const Ra_iterator<it1> &, const Ra_iterator <it2> &);

        template <typename it1, typename it2>
        friend bool operator>(const Ra_iterator<it1> &, const Ra_iterator <it2> &);
        
        template <typename it1, typename it2>
        friend bool operator<(const Ra_iterator<it1> &, const Ra_iterator <it2> &);

        /** ************************************************************************** */
		/**                                ACCESS OPERATORS                            */
		/** ************************************************************************** */

        reference operator*()
        {
            return *it;
        }

        pointer operator->() const
        {
            return it;
        }
        Ra_iterator &operator++()
        {
            ++it;
            return *this;
        }
        Ra_iterator &operator--()
        {
            --it;
            return *this;
        }
        Ra_iterator operator++(int)
        {
            Ra_iterator copie(*this);
            ++it;
            return copie;
        }
        Ra_iterator operator--(int)
        {
            Ra_iterator copie(*this);
            --it;
            return copie;
        }

        /** ************************************************************************** */
		/**                                ARETHMETIC OPERATORS                        */
		/** ************************************************************************** */


        Ra_iterator operator+(difference_type n)
        {
            return (this->it + n);
        }
        
        Ra_iterator operator-(difference_type n)
        {
            return (this->it - n);
        }

        Ra_iterator &operator+=(difference_type n)
        {
            it += n;
            return (*this);
        }

        Ra_iterator &operator-=(difference_type n)
        {
            it -= n;
            return (*this);
        }

        reference operator[](difference_type n)
        {
            return (it[n]);
        }
        template<class it1, class it2>
        friend typename Ra_iterator<it1>::difference_type operator-(const Ra_iterator<it1> &cp1, const Ra_iterator<it2> &cp2);
		template<class it1>
		friend Ra_iterator<it1> operator+(typename Ra_iterator<it1>::difference_type n, const Ra_iterator<it1> &cp);
        
    }; /// RANDOM_ACCESS_ITERATOR class

    /** ************************************************************************** */
	/**                                COMPARAISON OPERATORS DEFINITION            */
	/** ************************************************************************** */
    
    template <typename it1, typename it2>
    bool operator==(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it == s1.it);
    }

    template <typename it1, typename it2>
    bool operator!=(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it != s1.it);
    }

    template <typename it1, typename it2>
    bool operator>=(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it >= s1.it);
    }

    template <typename it1, typename it2>
    bool operator<=(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it <= s1.it);
    }

    template <typename it1, typename it2>
    bool operator>(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it > s1.it);
    }
 
    template <typename it1, typename it2>
    bool operator<(const Ra_iterator<it1> &s, const Ra_iterator <it2> &s1)
    {
        return (s.it < s1.it);
    }

    template<class it1, class it2>
    typename Ra_iterator<it1>::difference_type operator-(const Ra_iterator<it1> &cp1, const Ra_iterator<it2> &cp2)
	{
        return (cp1.base() - cp2.base());
    }
    template<class it1>
	Ra_iterator<it1> operator+(typename Ra_iterator<it1>::difference_type n, const Ra_iterator<it1> &cp)
    {
        return (Ra_iterator<it1>(cp.base() + n));
    }
} // for namespace
#endif