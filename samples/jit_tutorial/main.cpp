/*******************************************************************************
 *
 *   JIT tutorial sample: branching and graph reuse.
 *
 *   Demonstrates:
 *   - Tape: re-records and therefore follows normal C++ control flow per run.
 *   - JIT: graph is recorded once; plain C++ `if` is baked in at record time.
 *   - ABool::If: records a conditional node so the branch can vary at runtime.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2025 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Affero General Public License for more details.
 *
 *   You should have received a copy of the GNU Affero General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <XAD/XAD.hpp>

#include <iostream>

namespace
{

template <class AD>
AD piecewise_plain_if(const AD& x)
{
    // Normal C++ control flow: decision is made immediately, based on the current value.
    if (xad::value(x) < 2.0)
        return 1.0 * x;
    return 7.0 * x;
}

template <class AD>
AD piecewise_abool_if(const AD& x)
{
    // Trackable control flow for JIT: record both branches and select at runtime.
    auto cond = xad::less(x, 2.0);
    AD t = 1.0 * x;
    AD f = 7.0 * x;
    return cond.If(t, f);
}

}  // namespace

int main()
{
    std::cout << "Comparing Tape vs. JIT for the following two functions (inputs: x=1, x=3)\n";
    std::cout << "f1(x) = (x < 2) ? (1*x) : (7*x)          (plain C++ if)\n";
    std::cout << "f2(x) = less(x,2).If(1*x, 7*x)           (ABool::If)\n";

    // -------------------------------------------------------------------------
    // 1) Tape using f1 (plain if): re-records each run, so control flow is evaluated per input.
    // -------------------------------------------------------------------------
    {
        using mode = xad::adj<double>;
        using tape_type = mode::tape_type;
        using AD = mode::active_type;

        std::cout << "\n1) Tape using f1:\n";
        int runNo = 0;
        auto run = [&runNo](double x0) {
            ++runNo;
            tape_type tape;
            AD x = x0;
            tape.registerInput(x);

            tape.newRecording();
            AD y = piecewise_plain_if(x);  // f1
            tape.registerOutput(y);
            xad::derivative(y) = 1.0;
            tape.computeAdjoints();

            std::cout << "Tape run " << runNo << "   input: x=" << x0 << "  result:  y=" << xad::value(y)
                      << "  dy/dx=" << xad::derivative(x) << "\n";
        };

        run(1.0);  // y=1,  dy/dx=1
        run(3.0);  // y=21, dy/dx=7
    }

    // -------------------------------------------------------------------------
    // 2) JIT using f1 (plain if): record once; plain if is baked in at record time.
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;
        jit.registerInput(x);

        std::cout << "\n2) JIT using f1:\n";
        std::cout << "JIT plain-if: record at x=1, replay at x=3 (expected fail)\n";
        AD y = piecewise_plain_if(x);  // f1
        jit.registerOutput(y);
        jit.compile();

        double out = 0.0;
        jit.forward(&out, 1);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT recording with input:  x=1  y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";

        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out, 1);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=3  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot())
                  << "  (expected y=21, dy/dx=7)\n";
    }

    // -------------------------------------------------------------------------
    // 3) Tape using f2 (ABool::If): still works fine (ABool is passive when not JIT-recording).
    // -------------------------------------------------------------------------
    {
        using mode = xad::adj<double>;
        using tape_type = mode::tape_type;
        using AD = mode::active_type;

        std::cout << "\n3) Tape using f2:\n";
        int runNo = 0;
        auto run = [&runNo](double x0) {
            ++runNo;
            tape_type tape;
            AD x = x0;
            tape.registerInput(x);

            tape.newRecording();
            AD y = piecewise_abool_if(x);  // f2
            tape.registerOutput(y);
            xad::derivative(y) = 1.0;
            tape.computeAdjoints();

            std::cout << "Tape run " << runNo << "   input: x=" << x0 << "  result:  y=" << xad::value(y)
                      << "  dy/dx=" << xad::derivative(x) << "\n";
        };

        run(1.0);
        run(3.0);
    }

    // -------------------------------------------------------------------------
    // 4) JIT using f2 (ABool::If): records a conditional node, so branch varies per replay.
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;
        jit.registerInput(x);

        std::cout << "\n4) JIT using f2:\n";
        std::cout << "JIT ABool.If: record once, replay at x=1 and x=3 (expected ok)\n";
        AD y = piecewise_abool_if(x);  // f2
        jit.registerOutput(y);
        jit.compile();

        double out = 0.0;

        x = 1.0;
        jit.clearDerivatives();
        jit.forward(&out, 1);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=1  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";

        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out, 1);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=3  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
    }

    return 0;
}


