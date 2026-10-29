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
 *   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
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

#include <iomanip>
#include <iostream>
#include <vector>

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
    // xad::less returns an ABool (trackable boolean) instead of a plain bool.
    // Note: ABool also works with Tape mode (see example 3 below) - it converts to bool
    // and If() falls back to passive selection.
    auto cond = xad::less(x, 2.0);  // returns xad::ABool
    AD t = 1.0 * x;
    AD f = 7.0 * x;
    return cond.If(t, f);
}

}  // namespace

int main()
{
    std::cout << "Comparing Tape vs. JIT for the following two functions\n";
    std::cout << "f1(x) = (x < 2) ? (1*x) : (7*x)          (plain C++ if)\n";
    std::cout << "f2(x) = less(x,2).If(1*x, 7*x)           (ABool::If)\n";
    std::cout << "(f2 is semantically the same as f1, but expressed in a way JIT can record as a conditional)\n";
    std::cout << "\n";
    std::cout << "Example settings:\n";
    std::cout << "Tape: run 1 uses x=1, run 2 uses x=3 (re-records per run)\n";
    std::cout << "JIT : record uses x=1, replay uses x=3 (same recorded graph)\n";

    struct Row
    {
        const char* scenario;
        double x;
        double y;
        double dydx;
        const char* note;
    };
    std::vector<Row> rows;

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

            return std::make_pair(xad::value(y), xad::derivative(x));
        };

        auto r1 = run(1.0);  // y=1,  dy/dx=1
        auto r2 = run(3.0);  // y=21, dy/dx=7
        rows.push_back({"Tape f1", 1.0, r1.first, r1.second, ""});
        rows.push_back({"Tape f1", 3.0, r2.first, r2.second, ""});
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
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT recording with input:  x=1  y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT f1 (record)", 1.0, out, jit.getDerivative(x.getSlot()), ""});

        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=3  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot())
                  << "  (expected y=21, dy/dx=7)\n";
        rows.push_back({"JIT f1 (replay)", 3.0, out, jit.getDerivative(x.getSlot()), "expected fail"});
    }

    // -------------------------------------------------------------------------
    // 3) Tape using f2 (ABool::If): still works fine (ABool is passive when not JIT-recording).
    // -------------------------------------------------------------------------
    {
        using mode = xad::adj<double>;
        using tape_type = mode::tape_type;
        using AD = mode::active_type;

        std::cout << "\n3) Tape using f2: (works fine; ABool is passive when not JIT-recording)\n";
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

            return std::make_pair(xad::value(y), xad::derivative(x));
        };

        auto r1 = run(1.0);
        auto r2 = run(3.0);
        rows.push_back({"Tape f2", 1.0, r1.first, r1.second, ""});
        rows.push_back({"Tape f2", 3.0, r2.first, r2.second, "Tape supports ABool too"});
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
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=1  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT f2 (record)", 1.0, out, jit.getDerivative(x.getSlot()), ""});

        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "JIT run with input: x=3  result: y=" << out << "  dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT f2 (replay)", 3.0, out, jit.getDerivative(x.getSlot()), "replay picks correct branch"});
    }

    std::cout << "\nSummary:\n";
    std::cout << std::left << std::setw(22) << "Scenario"
              << std::right << std::setw(6) << "x"
              << std::setw(10) << "y"
              << std::setw(10) << "dy/dx"
              << "  " << "note"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    for (const auto& r : rows)
    {
        std::cout << std::left << std::setw(22) << r.scenario
                  << std::right << std::setw(6) << r.x
                  << std::setw(10) << r.y
                  << std::setw(10) << r.dydx
                  << "  " << r.note
                  << "\n";
    }

    return 0;
}


