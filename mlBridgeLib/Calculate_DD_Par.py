# Assume all ACBL double dummy and par calculations are wrong! Recompute both.

from mlBridgeLib.mlBridgeLib import NESW, vul_dds_d, NSHDC, seats
import dds
import ctypes
import functions


# valdiate calculated par result by comparing against known (assumed) correct par. Similar to functions.ComparePar().
def ComparePars(par1, par2):
    if par1[0] != par2[0]:
        return False
    if len(par1[1]) != len(par2[1]):
        return False
    for p1,p2 in zip(sorted(par1[1],key=lambda k: k[1]),sorted(par2[1],key=lambda k: k[1])):
        if p1 != p2:
            if p1[1] != p2[1] or p1[2] != p2[2] or p1[3] != p2[3]:
                return False # suit/double/direction difference
            if p1[0]+p1[4] != p2[0]+p2[4]:
                return False # only needs to have trick count same. ignore if levels are same (min level+overs vs max level+0)
    return True


# valdiate calculated dd result by comparing against known (assumed) correct dd. Similar to functions.CompareTable().
def CompareDDTables(DDtable1, DDtable2):
    #print(DDtable1,DDtable2)
    for suit in range(dds.DDS_STRAINS):
        for pl in range(4):
            if DDtable1[pl][suit] != DDtable2[pl][suit]:
                return False
    return True


# required df columns: PBN, Dealer, Vul
def Calculate_DD_Par(df, d):

    DDdealsPBN = dds.ddTableDealsPBN()
    tableRes = dds.ddTablesRes()
    pres = dds.allParResults()
    presmaster = dds.parResultsMaster()

    mode = 0
    tFilter = ctypes.c_int * dds.DDS_STRAINS
    trumpFilter = tFilter(0, 0, 0, 0, 0)
    line = ctypes.create_string_buffer(80)

    dds.SetMaxThreads(0)
    max_tables = dds.MAXNOOFTABLES
    nhands = len(df)
    
    for n,grp_start in enumerate(range(0,nhands,max_tables)):
        n += 1
        grp_count = min(nhands-grp_start,max_tables)
        DDdealsPBN.noOfTables = grp_count
        print(f"Processing group:{n} hands:{grp_start} to {grp_start+grp_count} of {nhands} dict len:{len(d)}")
        for handno in range(grp_count):
            r = df[['PBN','Dealer','Vul']].row(grp_start+handno)
            pbn, dealer, vul = r
            assert (pbn,dealer,vul) not in d, r
            # create lists of PBN
            pbn_encoded = pbn.encode() # requires pbn to be utf8 encoded.
            #print(len(pbn),pbn,pbn_encoded)
            DDdealsPBN.deals[handno].cards = pbn_encoded

        # CalcAllTablesPBN will do multi-threading
        res = dds.CalcAllTablesPBN(ctypes.pointer(DDdealsPBN), mode, trumpFilter, ctypes.pointer(tableRes), ctypes.pointer(pres))

        if res != dds.RETURN_NO_FAULT:
            dds.ErrorMessage(res, line)
            print(f"CalcAllTablesPBN: DDS error: {line.value.decode('utf-8')}")
            assert False, grp_start

        for handno in range(grp_count):
            r = df[['PBN','Dealer','Vul']].row(grp_start+handno)
            pbn, dealer, vul = r
            # compute double dummy
            assert dealer in NESW, r
            assert vul in ['None','N_S','E_W','Both'], r
            par_result = ctypes.pointer(presmaster)
            dd_result = ctypes.pointer(tableRes.results[handno])

            # Par calculations are not multi-threading
            res = dds.DealerParBin(dd_result, par_result, NESW.index(dealer), vul_dds_d[vul])
            if res != dds.RETURN_NO_FAULT:
                dds.ErrorMessage(res, line)

                print(f"DealerParBin: DDS error: {line.value.decode('utf-8')}")
                assert False, r

            DDtable_solved = tuple(tuple(dd_result.contents.resTable[suit][pl] for suit in [3,2,1,0,4]) for pl in range(4))

            score = par_result.contents.score
            if score == 0:
                par_solved = (0, [(0, '', '', '', 0)]) # par score is for everyone to pass (1 out of 100,000)
            else:
                assert par_result.contents.number > 0, r
                par_solved = (score,[])
                for i in range(par_result.contents.number):
                    ct = par_result.contents.contracts[i]    
                    #print(f"Par[{i}]: underTricks:{ct.underTricks} overTricks:{ct.overTricks} level:{ct.level} denom:{ct.denom} seats:{ct.seats}")
                    assert ct.underTricks == 0 or ct.overTricks == 0
                    par_solved[1].append((ct.level,NSHDC[ct.denom],'*' if ct.underTricks else '',seats[ct.seats],ct.overTricks-ct.underTricks))

            DDtable = DDtable_solved
            assert isinstance(DDtable,tuple), type(DDtable)
            assert len(DDtable)==4, DDtable
            assert all([isinstance(t,tuple) and len(t)==5 for t in DDtable]), DDtable

            par = par_solved
            assert type(par) is tuple, r
            assert len(par) == 2, r
            assert type(par[0]) is int, r
            assert type(par[1]) is list, r
            assert len(par[1]) > 0, r

            d[(pbn, dealer, vul)] = {'DDmakes':DDtable_solved,'Par':par_solved}
