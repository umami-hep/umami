MC Samples
==============

The FTAG1 derivations and the most recent ntuples for PFlow with the new RNNIP, SMT and the latest DL1* recommendations inside are shown in the following table. DIPS Default, DIPS Loose, DL1d Default, DL1d Loose and UMAMI are added as DL2 in the h5 ntuples.

A detailed list with the available derivations can be fund in the [FTAG algorithm documentation](https://ftag.docs.cern.ch/algorithms/overview/#training-samples).




## Release 22

### Release 22 Samples with Muons

The round 2 release 22 samples with RNNIP, DL1* and DIPS. Muon information are added (softMuon). Information for GNN training is added. The default and loose track selections are added. Default tracks are called `tracks` and loose tracks are called `tracks_loose`.

| Sample | h5 ntuples | DAOD_PHYSVAL derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
| ttbar | user.alfroch.410470.btagTraining.e6337_e5984_s3126_r12629_p4724.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSVAL.e6337_e5984_s3126_r12629_p4724 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.AOD.e6337_e5984_s3126_r12629
| Z' Extended (With QSP, Yes shower weights) | user.alfroch.800030.btagTraining.e7954_s3672_r12629_r12636_p4724.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3672_r12629_r12636_p4724 |  |
| Z' | user.alfroch.500567.btagTraining.e7954_e7400_s3672_r12629_r12636_p4724.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.500567.MGH7EG_NNPDF23ME_Zprime.deriv.DAOD_PHYSVAL.e7954_e7400_s3672_r12629_r12636_p4724 | mc16_13TeV.500567.MGH7EG_NNPDF23ME_Zprime.merge.AOD.e7954_e7400_s3672_r12629_r12636 |




??? info "Release 22 - Round 2 Samples"



    The Round 2 release 22 samples with RNNIP, DL1* and DIPS. The default and loose track selections are added. Default tracks are called `tracks` and loose tracks are called `tracks_loose`.

    | Sample | h5 ntuples | DAOD_PHYSVAL derivations| AOD |
    | ------------- | ---------------- | ---------------- | ---------------- |
    | ttbar | user.alfroch.410470.btagTraining.e6337_e5984_s3126_r12629_p4567.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSVAL.e6337_e5984_s3126_r12629_p4567 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.AOD.e6337_e5984_s3126_r12629
    | Z' Extended (With QSP, Yes shower weights) | uuser.alfroch.800030.btagTraining.e7954_s3672_r12629_p4567.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3672_r12629_p4567 | |
    | Z' | user.alfroch.500567.btagTraining.e7954_s3672_r12629_p4567.EMPFlow.2021-11-22-T133600-R31908_output.h5 | mc16_13TeV.500567.MGH7EG_NNPDF23ME_Zprime.deriv.DAOD_PHYSVAL.e7954_s3672_r12629_p4567 | |






??? info "Small Validation Release 22 Samples"


    The small validation release 22 samples with RNNIP, DL1* and DIPS.

    | Sample | h5 ntuples | h5 ntuples (looser track selection)   |  DAOD_PHYSVAL derivations| AOD |
    | ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
    |ttbar | user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSVAL.e6337_s3126_r12305_r12253_r12305_p4505 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.merge.AOD.e6337_e5984_s3126_r12305_r12253_r12305_r12298 |
    |Z' Extended (With QSP, Yes shower weights) | user.alfroch.800030.btagTraining.e7954_s3582_r12305_r12253_r12305_r12298_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.800030.btagTraining.e7954_s3582_r12305_r12253_r12305_r12298_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3582_r12305_r12253_r12305_r12298_p4505 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.merge.AOD.e7954_s3582_r12305_r12253_r12305_r12298 |
    |Z' Extended (No QSP, Yes shower weights) | user.alfroch.800030.btagTraining.e7954_s3126_r12305_r12253_r12305_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.800030.btagTraining.e7954_s3126_r12305_r12253_r12305_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3126_r12305_r12253_r12305_p4505 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.merge.AOD.e7954_e7400_s3126_r12305_r12253_r12305_r12298 |




---

## Release 21

### Release 21 - Default FTAG Samples (ttbar and Z')

| Sample | h5 ntuples | h5 ntuples (looser track selection) |  FTAG1 derivations | AOD |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
|MC16a - ttbar | user.alfroch.410470.btagTraining.e6337_s3126_r9364_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.410470.btagTraining.e6337_s3126_r9364_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r9364_p3985 | |
|MC16a - Z' | user.alfroch.427080.btagTraining.e5362_s3126_r9364_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.427080.btagTraining.e5362_s3126_r9364_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r9364_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r9364 |
|MC16d - ttbar | user.alfroch.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p3985 | |
|MC16d - Z' | user.alfroch.427080.btagTraining.e5362_s3126_r10201_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.427080.btagTraining.e5362_s3126_r10201_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10201_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10201 |
|MC16d - Z' extended | user.alfroch.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.recon.AOD.e6928_s3126_r10201 |
|MC16d - Z' extended  (QSP on)| user.alfroch.800030.btagTraining.e7954_e7400_s3663_r10201_p4207.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.800030.btagTraining.e7954_e7400_s3663_r10201_p4207.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e7954_e7400_s3663_r10201_p4207 ||
|MC16e - ttbar | user.alfroch.410470.btagTraining.e6337_s3126_r10724_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.410470.btagTraining.e6337_s3126_r10724_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10724_p3985 | |
|MC16e - Z' | user.alfroch.427080.btagTraining.e5362_s3126_r10724_p3985.EMPFlow.2021-09-07-T122808-R14883_output.h5 | user.alfroch.427080.btagTraining.e5362_s3126_r10724_p3985.EMPFlow_loose.2021-09-07-T122950-R13989_output.h5 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10724_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10724 |




The Z' & Z+jets FTAG 1 derivations were requested [here](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)


### Z+jets Samples for bb category


??? info "MC16d"



    For MC16d the p-tag p3985 was [requested](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)

    In addition there are the Z+jets (Z->mumu/nunu) samples for the bb extension

    | Sample | h5 ntuples    |  FTAG1 derivations| AOD |
    | ------------- | ---------------- | ---------------- | ---------------- |
    |Znunu | user.mguth.366010.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210   |
    |Znunu | user.mguth.366011.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |Znunu | user.mguth.366012.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |Znunu | user.mguth.366013.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |Znunu | user.mguth.366014.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |Znunu | user.mguth.366015.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210  |
    |Znunu | user.mguth.366016.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |Znunu | user.mguth.366017.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
    |||||
    |Zmumu | user.mguth.364102.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
    |Zmumu | user.mguth.364105.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.merge.AOD.e5271_s3126_r10201_r10210  |
    |Zmumu | user.mguth.364108.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.merge.AOD.e5271_s3126_r10201_r10210       |
    |Zmumu | user.mguth.364111.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
    |Zmumu | user.mguth.364112.btagTraining.e5271_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.merge.AOD.e5271_e5984_s3126_r10201_r10210 |
    |Zmumu | user.mguth.364113.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.merge.AOD.e5271_s3126_r10201_r10210 |




??? info "MC16a"


    [Derivation request](https://prodtask-dev.cern.ch/prodtask/inputlist_with_request/30189/) for p4062

    | Sample | h5 ntuples    |  FTAG1 derivations| AOD |
    | ------------- | ---------------- | ---------------- | ---------------- |
    |Znunu |  | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.recon.AOD.e6695_s3126_r9364|
    |Znunu |  | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r9364 |
    |Znunu |  | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r9364 |
    |Znunu |  | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r9364 |
    |Znunu |  | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r9364 |
    |Znunu |  | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r9364  |
    |Znunu |  | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r9364 |
    |Znunu |  | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.recon.AOD.e6695_s3126_r9364 |
    |||||
    |Zmumu |  | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.recon.AOD.e5271_s3126_r9364 |
    |Zmumu |  | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.recon.AOD.e5271_s3126_r9364 |
    |Zmumu |  | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.recon.AOD.e5271_s3126_r9364 |
    |Zmumu |  | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.recon.AOD.e5271_s3126_r9364 |
    |Zmumu |  | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.recon.AOD.e5271_s3126_r9364 |
    |Zmumu |  | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.recon.AOD.e5271_s3126_r9364 |




??? info "MC16e"

    [Derivation request](https://prodtask-dev.cern.ch/prodtask/inputlist_with_request/30189/) for p4062

    | Sample | h5 ntuples    |  FTAG1 derivations| AOD |
    | ------------- | ---------------- | ---------------- | ---------------- |
    |Znunu |  | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.recon.AOD.e6695_s3126_r10724|
    |Znunu |  | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r10724 |
    |Znunu |  | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r10724 |
    |Znunu |  | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r10724 |
    |Znunu |  | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r10724 |
    |Znunu |  | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r10724  |
    |Znunu |  | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r10724 |
    |Znunu |  | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.recon.AOD.e6695_s3126_r10724 |
    |||||
    |Zmumu |  | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.recon.AOD.e5271_s3126_r10724 |
    |Zmumu |  | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.recon.AOD.e5271_s3126_r10724 |
    |Zmumu |  | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.recon.AOD.e5271_s3126_r10724 |
    |Zmumu |  | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.recon.AOD.e5271_s3126_r10724 |
    |Zmumu |  | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.recon.AOD.e5271_s3126_r10724 |
    |Zmumu |  | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.recon.AOD.e5271_s3126_r10724 |

