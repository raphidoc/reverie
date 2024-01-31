import pandas as pd


def tidy_6s_output(wls, res):
    # Iterate over the wavelength and result, store dataframes rows in a list
    wls_dfs = []
    for wl, r in zip(wls, res):
        res_dic = r.__dict__

        # iterate over the values of the results
        dfs = []
        for k in res_dic:
            # Transform non nested values to DF
            if k == "values":
                dfs.append(pd.DataFrame(res_dic[k], index=[0]))
                continue

            # Unnest transmittance values
            if k == "trans":
                trans = pd.DataFrame()
                for x, y in res_dic[k].items():
                    down = f"{x}_downward"
                    up = f"{x}_upward"
                    tot = f"{x}_total"

                    dic = {down: y.downward, up: y.upward, tot: y.total}

                    # pd.DataFrame(dic, index=[0])

                    trans = pd.concat(
                        [trans, pd.DataFrame(dic, index=[0])],
                        axis=1,
                        ignore_index=False,
                    )

                dfs.append(trans)
                continue

            # Unnest Rayleigh aerosol values
            if k == "rat":
                rat = pd.DataFrame()
                for x, y in res_dic[k].items():
                    down = f"{x}_rayleigh"
                    up = f"{x}_aerosol"
                    tot = f"{x}_total"

                    dic = {down: y.rayleigh, up: y.aerosol, tot: y.total}

                    rat = pd.concat(
                        [rat, pd.DataFrame(dic, index=[0])], axis=1, ignore_index=False
                    )

                dfs.append(rat)
                continue

            # Pass 6S text output as a dataframe column
            if k == "fulltext":
                dfs.append(pd.DataFrame({"fulltext": res_dic["fulltext"]}, index=[0]))
                continue

        wl_df = pd.concat(dfs, axis=1, ignore_index=False)
        wl_df.insert(0, "wavelength", wl)
        wls_dfs.append(wl_df)

    wls_df = pd.concat(wls_dfs, axis=0, ignore_index=True)

    # Optional: add UUID to the dataframe
    # UUID = np.unique(MatchupDF["UUID"])
    # wls_df.insert(0, "UUID", MatchupDF["UUID"].values, True)

    return wls_df
