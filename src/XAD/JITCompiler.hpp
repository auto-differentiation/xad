#pragma once

#include <XAD/Config.hpp>
#include <XAD/Exceptions.hpp>
#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>
#include <complex>
#include <vector>

namespace xad
{

template <class Scalar, std::size_t M>
struct AReal;

template <class Real, std::size_t N = 1>
class JITCompiler
{
  public:
    typedef unsigned int size_type;
    typedef unsigned int slot_type;
    typedef slot_type position_type;
    typedef AReal<Real, N> active_type;
    typedef Real value_type;
    typedef JITCompiler<Real, N> jit_type;
    typedef typename DerivativesTraits<Real, N>::type derivative_type;

    static constexpr slot_type INVALID_SLOT = slot_type(-1);

    explicit JITCompiler(bool activate = true)
    {
        if (activate)
            setActive(this);
    }

    ~JITCompiler()
    {
        deactivate();
    }

    JITCompiler(JITCompiler&& other) noexcept
    {
        if (other.isActive())
        {
            other.deactivate();
            setActive(this);
        }
    }

    JITCompiler& operator=(JITCompiler&& other) noexcept
    {
        if (this != &other)
        {
            deactivate();
            if (other.isActive())
            {
                other.deactivate();
                setActive(this);
            }
        }
        return *this;
    }

    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;

    XAD_INLINE void activate() { setActive(this); }

    XAD_INLINE void deactivate()
    {
        if (active_jit_ == this)
            active_jit_ = nullptr;
    }

    XAD_INLINE bool isActive() const { return active_jit_ == this; }
    XAD_INLINE static JITCompiler* getActive() { return active_jit_; }

    XAD_INLINE static void setActive(JITCompiler* j)
    {
        if (active_jit_ != nullptr)
            throw OutOfRange("JIT Compiler already active");
        else
            active_jit_ = j;
    }

    XAD_INLINE static void deactivateAll() { active_jit_ = nullptr; }

    XAD_INLINE void registerInput(active_type& inp)
    {
        if (!inp.shouldRecord())
        {
            inp.slot_ = registerVariable();
            pushLhs(inp.slot_);
        }
    }

    XAD_INLINE void registerInput(std::complex<active_type>& inp)
    {
        auto reim_ptr = reinterpret_cast<active_type*>(&inp);
        registerInput(reim_ptr[0]);
        registerInput(reim_ptr[1]);
    }

    XAD_INLINE void registerOutput(active_type& outp)
    {
        if (!outp.shouldRecord())
        {
            outp.slot_ = registerVariable();
            pushLhs(outp.slot_);
        }
    }

    XAD_INLINE void registerOutput(std::complex<active_type>& outp)
    {
        auto reim_ptr = reinterpret_cast<active_type*>(&outp);
        registerOutput(reim_ptr[0]);
        registerOutput(reim_ptr[1]);
    }

    template <class Inner>
    XAD_INLINE void registerInputs(std::vector<Inner>& v)
    {
        registerInputs(v.begin(), v.end());
    }

    template <class It>
    XAD_INLINE void registerInputs(It first, It last)
    {
        while (first != last) registerInput(*first++);
    }

    XAD_INLINE void registerOutputs(std::vector<active_type>& v)
    {
        for (auto& x : v) registerOutput(x);
    }

    template <class It>
    XAD_INLINE void registerOutputs(It first, It last)
    {
        while (first != last) registerOutput(*first++);
    }

    void newRecording() { /* later */ }
    void computeAdjoints() { /* later */ }
    void clearAll() { /* later */ }

    void clearDerivatives() { /* later */ }

    derivative_type& derivative(slot_type s)
    {
        static derivative_type dummy = derivative_type();
        return dummy;
    }

    const derivative_type& derivative(slot_type s) const
    {
        static derivative_type dummy = derivative_type();
        return dummy;
    }

    derivative_type getDerivative(slot_type s) const { return derivative_type(); }
    void setDerivative(slot_type s, const derivative_type& d) { /* later */ }
    void setDerivative(slot_type s, derivative_type&& d) { /* later */ }

    slot_type registerVariable()
    {
        return slot_counter_++;
    }

    XAD_INLINE void pushLhs(slot_type) { /* later */ }

    template <class MulIt, class SlotIt>
    XAD_FORCE_INLINE void pushAll(MulIt, SlotIt, unsigned) { /* later */ }

    void printStatus() const { /* later */ }
    std::size_t getMemory() const { return 0; }

    position_type getPosition() const { return 0; }
    void clearDerivativesAfter(position_type) { /* later */ }
    void resetTo(position_type) { /* later */ }
    void computeAdjointsTo(position_type) { /* later */ }

  private:
    static XAD_THREAD_LOCAL JITCompiler* active_jit_;
    slot_type slot_counter_ = 0;
};

template <class Real, std::size_t N>
XAD_THREAD_LOCAL JITCompiler<Real, N>* JITCompiler<Real, N>::active_jit_ = nullptr;

}  // namespace xad
