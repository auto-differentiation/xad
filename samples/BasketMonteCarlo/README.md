# BasketMonteCarlo

Prices a European call on a basket of 12 correlated assets by Monte Carlo. In the same
run it computes the pathwise delta and vega of every asset in adjoint mode, with the
paths split across threads. Each thread has its own tape and records one path at a time,
so the tape memory stays the same however many paths there are.

The market data is fixed and no data file is needed:

- 12 assets, equally weighted, with spots between `90` and `110` and volatilities between
  `15%` and `35%`
- the same correlation of `0.5` between every pair of assets
- strike `100`, rate `3%` and maturity `1` year

## How it works

**Model.** For each path the sample draws independent standard normals `eps`. It
correlates them with the Cholesky factor `L` of the correlation matrix, `z = L eps`, and
moves each asset to maturity:

    S_i(T) = S_i * exp((r - vol_i^2 / 2) T + vol_i sqrt(T) z_i)

The discounted payoff is `exp(-r T) * max(sum_i w_i S_i(T) - K, 0)`. The pricer for one
path is written once as a template, so the same code runs with `double` and with XAD's
active type. The random numbers and `L` stay in `double`, since the Greeks are taken
with respect to the spots and volatilities only.

**Pathwise Greeks.** Delta and vega are the averages over the paths of the derivatives
of each path's payoff. On a path that ends in the money,

    dS_i(T)/dS_i   = S_i(T) / S_i
    dS_i(T)/dvol_i = S_i(T) * (-vol_i T + sqrt(T) z_i)

times `exp(-r T) w_i`. A path out of the money contributes zero. The kink at the strike
is hit with probability zero, so averaging these derivatives gives the derivatives of the
price. XAD computes all 24 of them for a path with one adjoint sweep.

**One tape per thread, one path per recording.** XAD's active tape is thread-local, and
a tape becomes active on the thread that constructs it. So each worker thread constructs
its own tape at the start. For every path the worker then:

1. clears the tape with `clearAll()`;
2. registers the spots and volatilities as inputs and starts a new recording;
3. prices the path and runs the adjoint sweep;
4. adds the payoff and the input adjoints to its sums.

The tape never holds more than one path: about 1 KiB here, for any number of paths.
This is the same per-path pattern as in [`samples/LiborSwaptionPricer`](../LiborSwaptionPricer),
spread across threads.

**Same results on any number of threads.** The paths are grouped into fixed chunks of 256.
Each chunk has its own random number stream, seeded from the fixed seed and the chunk
number, and its own entry for its sums. The threads share out the chunks, and at the end
the chunk sums are added up in chunk order. So every chunk does exactly the same
operations, in the same order, whichever thread runs it. The result is bitwise the same
for 1 thread and for any other number.

The random numbers come from a small splitmix64 generator with the Box-Muller transform,
rather than from `<random>`. `std::normal_distribution` gives different numbers with
different standard libraries.

If XAD is built with `XAD_NO_THREADLOCAL`, only one tape can be active in the whole
process. The sample then runs on 1 thread and says so.

## Checks

The sample returns a non-zero exit code if any of these fails:

- **Price.** The price from the XAD run matches the plain `double` pricing to a relative
  `1e-12`.
- **Greeks.** Bump-and-revalue gives every delta and vega with central differences. It
  uses the same random numbers, so it differs from XAD only by the bumping error, and the
  Greeks must match it to a relative `1e-5`. The bumps are small (`1e-6` relative for the
  spots and absolute for the volatilities). A path whose basket ends within a bump of the
  strike crosses the kink, where the difference quotient is not the slope. That error
  grows with the bump size, so with bumps of `1e-4` the differences are about `2e-5`. XAD's
  derivatives have no such error.
- **Threads.** When more than one thread is used, the XAD run is repeated on 1 thread and
  must give exactly the same price and Greeks.

## Cost

The sample reports the cost of all Greeks as a multiple of one plain pricing, as absolute
times depend on the machine. All three runs use the same number of threads.

Bump-and-revalue needs `4 * 12 + 1 = 49` pricings of each path for 24 Greeks, and its
cost grows with the number of inputs. The adjoint sweep gives all of them for a cost of a
few pricings, whatever their number. Here bumping costs less than 49 times the plain
pricing, because the random numbers and their correlation are generated once per path
and reused for every bump.

## Build and run

From the repository root:

    cmake -S . -B build
    cmake --build build --target BasketMonteCarlo
    build/samples/BasketMonteCarlo/BasketMonteCarlo [numPaths] [numThreads]

The defaults are 100,000 paths and as many threads as the hardware supports. As a test,
`ctest -R BasketMonteCarlo` runs 4,096 paths on 4 threads.

## Expected output

The cost multiples and the speed-up depend on the machine, and the last digits of the
relative differences vary between compilers and platforms. A representative run of
`BasketMonteCarlo 100000 8` in a release build prints:

    European basket call on 12 assets, equally weighted, correlation 0.50
    strike 100.00, rate 3.00%, maturity 1.00 years
    100000 paths on 8 thread(s), in 391 chunks of up to 256 paths

    price 9.0726941825 (plain), 9.0726941825 (XAD)

    asset     spot    vol   delta (XAD)  delta (bump)  rel. diff    vega (XAD)   vega (bump)  rel. diff
        0   100.00   0.20  0.0499570239  0.0499570239  3.394e-11  2.3011201020  2.3011201020  2.826e-12
        1    95.00   0.25  0.0512374835  0.0512374835  1.788e-11  2.2127911381  2.2127911381  2.518e-11
        2   105.00   0.18  0.0495302655  0.0495302655  4.731e-11  2.4481607457  2.4481607457  8.439e-12
        3    98.00   0.30  0.0524749707  0.0524749706  5.999e-11  2.2842127015  2.2842127014  4.319e-11
        4   102.00   0.22  0.0504573657  0.0504573657  1.851e-11  2.3608847622  2.3608847621  5.768e-12
        5   110.00   0.35  0.0538112881  0.0538112881  5.846e-11  2.5796052358  2.5796052357  3.330e-11
        6    90.00   0.15  0.0487960932  0.0487960932  2.361e-12  2.0902223008  2.0902223008  5.698e-12
        7   101.00   0.28  0.0520128836  0.0520128836  7.580e-12  2.3645965718  2.3645965717  1.331e-11
        8    97.00   0.24  0.0509985419  0.0509985419  3.483e-11  2.2669920100  2.2669920100  5.138e-12
        9   103.00   0.19  0.0497763908  0.0497763908  4.342e-11  2.4058901411  2.4058901411  1.188e-12
       10    99.00   0.32  0.0530000082  0.0530000082  5.674e-11  2.3167768890  2.3167768890  3.189e-11
       11   108.00   0.26  0.0515156822  0.0515156822  2.255e-11  2.5350152682  2.5350152681  2.719e-11

    cost relative to one plain pricing (24 inputs)
      price and all Greeks with XAD:     3.2x
      bump-and-revalue (central):        8.8x  (49 pricings per path)
    largest tape: 1.3 KiB per thread, whatever the number of paths

    XAD on 1 thread: identical results, 3.7x the time on 8 threads
