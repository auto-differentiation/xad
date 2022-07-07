/*******************************************************************************

   Unit tests for checkpointing

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>


template <class T>
void g(int n, T& x)
{
    for (int i = 0; i < n; ++i) x = sin(x);
}

template <class Tape>
class GCheckpointCallback : public xad::CheckpointCallback<Tape>
{
  public:
    typedef typename Tape::slot_type idx_type;

    void push(int n, double x, idx_type inputidx, idx_type outputidx)
    {
        n_.push(n);
        x_.push(x);
        idx_.push(inputidx);
        idx_.push(outputidx);
    }

    void computeAdjoint(Tape* tape) override
    {
        typedef typename Tape::active_type ad_type;

        // retrieve checkpoint variables
        idx_type outputidx = idx_.top();
        idx_.pop();
        idx_type inputidx = idx_.top();
        idx_.pop();  
        int n = n_.top();
        n_.pop();
        int c = int(x_.size());
        // these need to be combined for checkpointing - get output adjoint
        double outputder = tape->getAndResetOutputAdjoint(outputidx);

        // create new AD type and start nested recording from after that
        ad_type x = x_.top();
        x_.pop();
        tape->registerInput(x);

        xad::ScopedNestedRecording<Tape> nested(tape);

        g(n, x);

        std::cout << "ckpt " << c << ": " << tape->getMemory() << "\n";

        tape->registerOutput(x);
        derivative(x) = outputder;
        nested.computeAdjoints();
        nested.incrementAdjoint(inputidx, derivative(x));  // incr. input derivative
    }

    ~GCheckpointCallback()
    {
        assert(n_.empty());
        assert(x_.empty());
        assert(idx_.empty());
    }

  private:
    std::stack<int> n_;
    std::stack<double> x_;
    std::stack<idx_type> idx_;  // input, then output
};

template <class T>
void g_checkpointed(int n, T& x)
{
    typedef typename T::tape_type tape_type;
    typedef GCheckpointCallback<tape_type> callback_type;

    tape_type* tape = x.getTape();
    double xp = value(x);
    auto inslot = x.getSlot();
    double xin = xp;  // keep input for checkpoint

    // check if we have a cb object from before that we can push to
    callback_type* cb = nullptr;
    if (!tape->haveCallbacks())
    {
        cb = new GCheckpointCallback<typename T::tape_type>;
        tape->pushCallback(cb);
    }
    else
    {
        cb = static_cast<callback_type*>(tape->getLastCallback());
    }

    g(n, xp);

    // these 2 need to be combined - register output or sth.
    value(x) = xp;
    auto outslot = x.getSlot();
    cb->push(n, xin, inslot, outslot);  // add info for chkpt

    // register the callback object for the reverse path (with the checkpoint info)
    tape->insertCallback(cb);
}

template <class T>
void f(int n, int m, T& x)
{
    for (int i = 0; i < n; i += m)
    {
        g_checkpointed(std::min(m, n - i), x);
    }
}

size_t driver_adj(int n, int m, double& xv, double& xa)
{
    typedef xad::AReal<double> ad_type;
    typedef xad::Tape<double> tape_type;

    tape_type t;
    ad_type x = xv;
    t.registerInput(x);
    t.newRecording();

    f(n, m, x);

    std::cout << "ckpt 0: " << t.getMemory() << "\n";
    size_t ret = t.getMemory();

    t.registerOutput(x);
    derivative(x) = xa;
    t.computeAdjoints();

    xv = value(x);
    xa = derivative(x);
    return ret;
}

size_t driver_adj_nochkpt(int n, int m, double& xv, double& xa)
{
    typedef xad::AReal<double> ad_type;
    typedef xad::Tape<double> tape_type;
    (void)m;

    tape_type t;
    ad_type x = xv;
    t.registerInput(x);
    t.newRecording();

    g(n, x);

    std::cout << "r mem: " << t.getMemory() << "\n";
    size_t ret = t.getMemory();

    t.registerOutput(x);
    derivative(x) = xa;
    t.computeAdjoints();

    xv = value(x);
    xa = derivative(x);
    return ret;
}

TEST(Checkpointing, equidistantLoop)
{
    int n = 20, m = 4;
    double xv = 2.1, xa = 1.0;
    std::cout.precision(15);
    size_t memchkpt = driver_adj(n, m, xv, xa);
    // std::cout << "x = " << xv << std::endl;
    // std::cout << "xa = " << xa << std::endl;

    // std::cout << "\n\n-------------------------\n";

    // same without checkpointing
    double xv2 = 2.1, xa2 = 1.0;
    size_t memstraight = driver_adj_nochkpt(n, m, xv2, xa2);
    // std::cout << "x = " << xv2 << std::endl;
    // std::cout << "xa = " << xa2 << std::endl;

    EXPECT_EQ(xv2, xv);
    EXPECT_EQ(xa2, xa);
    EXPECT_LT(memchkpt, memstraight);
}

//#define VERBOSE 1
#define STATISTICS 1

size_t max_tape_size = 0;

enum RUN_MODE
{
    CHECKPOINT_ARGUMENTS_AND_RUN_PASSIVELY,
    GENERATE_TAPE
};

template <class T>
void g_rec_insert_checkpoint(int from, int to, int stride, T& x, RUN_MODE m = GENERATE_TAPE);

template <class T>
void g_rec(int from, int to, int stride, T& x)
{
    if (to - from > stride)
    {
        g_rec(from, from + (to - from) / 2, stride, x);
        g_rec(from + (to - from) / 2, to, stride, x);
    }
    else
        for (int i = from; i < to; i++) x = sin(x);
}

template <class Tape>
class GCheckpointCallback2 : public xad::CheckpointCallback<Tape>
{
  public:
    typedef typename Tape::value_type value_type;
    typedef typename Tape::slot_type idx_type;
    static std::stack<std::pair<int, value_type> > state;
    static int stride;
    GCheckpointCallback2() {}
    std::stack<int> fromto_;
    std::stack<idx_type> inout_;

    void computeAdjoint(Tape* tape) override
    {
        typedef typename Tape::active_type ad_type;

        // retrieve checkpoint variables
        int to = fromto_.top();
        fromto_.pop();
        int from = fromto_.top();
        fromto_.pop();
        idx_type outputidx = inout_.top();
        inout_.pop();
        idx_type inputidx = inout_.top();
        inout_.pop();

#ifdef VERBOSE
        std::cout << "top=" << state.top().second << std::endl;
        std::cout << "RESTORE CHECKPOINT FOR SECTION " << from << " ... " << to - 1 << std::endl;
#endif

        // these need to be combined for checkpointing - get output adjoint
        double outputder = tape->getAndResetOutputAdjoint(outputidx);

        // create new AD type and start nested recording from after that
        ad_type x = state.top().second;
        tape->registerInput(x);
        xad::ScopedNestedRecording<Tape> nested(tape);

        g_rec_insert_checkpoint(from, to, stride, x, GENERATE_TAPE);

        tape->registerOutput(x);
        derivative(x) = outputder;
#ifdef VERBOSE
        std::cout << "INTERPRET SECTION " << from << " ... " << to - 1 << std::endl;
#endif

        nested.computeAdjoints();
        nested.incrementAdjoint(inputidx, derivative(x));  // incr. input derivative
        if (to - from <= stride)
        {
#ifdef VERBOSE
            std::cout << "popping " << state.top().first << ", " << state.top().second << std::endl;
#endif
            state.pop();
        }
    }
};

template <class Tape>
int GCheckpointCallback2<Tape>::stride = 0;
template <class Tape>
std::stack<std::pair<int, typename Tape::value_type> > GCheckpointCallback2<Tape>::state;

template <class T>
void g_rec_insert_checkpoint(int from, int to, int stride, T& x, RUN_MODE m)
{
    typedef typename T::tape_type tape_type;
    typedef GCheckpointCallback2<tape_type> callback_type;
    tape_type* tape = x.getTape();

    if (m == CHECKPOINT_ARGUMENTS_AND_RUN_PASSIVELY)
    {
#ifdef VERBOSE
        std::cout << "STORE CHECKPOINT FOR SECTION " << from << " ... " << to - 1 << std::endl;
#endif

        // check if we have a cb object from before that we can push to
        callback_type* cb = nullptr;
        if (!tape->haveCallbacks())
        {
            cb = new GCheckpointCallback2<tape_type>;
            tape->pushCallback(cb);
        }
        else
        {
            cb = static_cast<callback_type*>(tape->getLastCallback());
        }
        double xv = value(x);
        auto inslot = x.getSlot();
        // double xin = xv;  // keep input for checkpoint
        cb->inout_.push(inslot);

        cb->fromto_.push(from);
        cb->fromto_.push(to);

        if (cb->state.empty() || from != cb->state.top().first)
        {
#ifdef VERBOSE
            std::cout << "PUSHING (" << from << ", " << xv << ")" << std::endl;
#endif

            cb->state.push(std::make_pair(from, xv));
        }

#ifdef VERBOSE
        std::cout << "RUN SECTION " << from << " ... " << to - 1 << " PASSIVELY" << std::endl;
#endif
        g_rec(from, to, stride, xv);

        tape->registerOutput(x);
        value(x) = xv;
        auto outslot = x.getSlot();
        cb->inout_.push(outslot);

        // register the callback object for the reverse path (with the checkpoint info)
        tape->insertCallback(cb);
    }
    else if (m == GENERATE_TAPE)
    {
#ifdef VERBOSE
        std::cout << "GENERATE TAPE FOR SECTION " << from << " ... " << to - 1 << std::endl;
#endif
        GCheckpointCallback2<tape_type>::stride = stride;
        if (to - from > stride)
        {
            g_rec_insert_checkpoint(from, from + (to - from) / 2, stride, x,
                           CHECKPOINT_ARGUMENTS_AND_RUN_PASSIVELY);
            g_rec_insert_checkpoint(from + (to - from) / 2, to, stride, x,
                           CHECKPOINT_ARGUMENTS_AND_RUN_PASSIVELY);
        }
        else
        {
            for (int i = from; i < to; ++i) x = sin(x);
        }
    }
#ifdef STATISTICS
    if (tape->getMemory() > max_tape_size)
        max_tape_size = tape->getMemory();
#endif
}

template <class T>
void f_rec(int from, int to, int stride, T& x)
{
    g_rec_insert_checkpoint(from, to, stride, x);
}

size_t driver_rec_adj(int n, int stride, double& xv, double& xa1)
{
    typedef xad::AReal<double> ad_type;
    typedef xad::Tape<double> tape_type;

    tape_type t;
    ad_type x = xv;
    t.registerInput(x);
    t.newRecording();

    f_rec(0, n, stride, x);

    t.registerOutput(x);
    derivative(x) = xa1;
    t.computeAdjoints();

    xv = value(x);
    xa1 = derivative(x);
    return max_tape_size;
}

TEST(Checkpointing, recursiveLoop)
{
    std::cout.precision(15);
    int n = 20, stride = 4;
    double x = 2.1, xa1 = 1.0;
    size_t memchkpt = driver_rec_adj(n, stride, x, xa1);
    // std::cout << "x=" << x << std::endl;
    // std::cout << "x_{(1)}=" << xa1 << std::endl;
    std::cout << "mem with checkpoint: " << memchkpt << std::endl;

    // same without checkpointing (same func as in other test above)
    double xv2 = 2.1, xa2 = 1.0;
    size_t memstraight = driver_adj_nochkpt(n, stride, xv2, xa2);

    // std::cout << "x = " << xv2 << std::endl;
    // std::cout << "xa = " << xa2 << std::endl;
    std::cout << "mem without checkpoint:" << memstraight << std::endl;

    EXPECT_EQ(xv2, x);
    EXPECT_EQ(xa2, xa1);
#if !defined(XAD_TAPE_REUSE_SLOTS) || defined(NDEBUG)
    // when we track re-usable slots, these increase memory of tape in debug builds for the slots,
    // due to more destructor calls / expression evaluations.
    // so we only check this for release builds
    EXPECT_LT(memchkpt, memstraight);
#endif
}
