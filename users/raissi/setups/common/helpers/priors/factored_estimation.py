
def get_diphone_priors(graphPath, model, dataPaths, datasetIndices,
                       nStateClasses=141, nContexts=47, gpu=1, time=20, isSilMapped=True, name=None, nBatch=10000, tf_library=None, tm=None):

    if tf_library is None:
        tf_library = libraryPath
    if tm is None:
        tm = defaultTfMap

    estimateJob = EstimateSprintDiphoneAndContextPriors(graphPath,
                                                        model,
                                                        dataPaths,
                                                        datasetIndices,
                                                        tf_library,
                                                        nContexts=nContexts,
                                                        nStateClasses=nStateClasses,
                                                        gpu=gpu,
                                                        time=time,
                                                        tensorMap=tm,
                                                        nBatch=nBatch ,)
    if name is not None:
        estimateJob.add_alias(f"priors/{name}")

    xmlJob = DumpXmlSprintForDiphone(estimateJob.diphoneFiles,
                                     estimateJob.contextFiles,
                                     estimateJob.numSegments,
                                     nContexts=nContexts,
                                     nStateClasses=nStateClasses,
                                     adjustSilence=isSilMapped)

    priorFiles = [xmlJob.diphoneXml, xmlJob.contextXml]

    xmlName = f"priors/{name}"
    tk.register_output(xmlName, priorFiles[0])

    return priorFiles



def get_triphone_priors(graphPath, model, dataPaths, nStateClasses=282, nContexts=47, nPhones=47, nStates=3,
                        cpu=2, gpu=1, time=1, nBatch=18000, dNum=3, sNum=20, step=200, dataOffset=10, segmentOffset=10,
                        name=None, tf_library=None, tm=None, isMulti=False):
    if tf_library is None:
        tf_library = libraryPath
    if tm is None:
        tm = defaultTfMap

    triphoneFiles = []
    diphoneFiles = []
    contextFiles = []
    numSegments = []


    for i in range(2, dNum + 2):
        startInd = i * dataOffset
        endInd = (i + 1) * dataOffset
        for j in range(sNum):
            startSegInd = j * segmentOffset
            endSegInd = (j + 1) * segmentOffset
            if endSegInd > 1248: endSegInd = 1248

            datasetIndices = list(range(startInd, endInd))
            estimateJob = EstimateSprintTriphonePriorsForward(graphPath,
                                                              model,
                                                              dataPaths,
                                                              datasetIndices,
                                                              startSegInd, endSegInd,
                                                              tf_library,
                                                              nContexts=nContexts,
                                                              nStateClasses=nStateClasses,
                                                              nStates=nStates,
                                                              nPhones=nPhones,
                                                              nBatch=nBatch,
                                                              cpu=cpu,
                                                              gpu=gpu,
                                                              time=time,
                                                              tensorMap=tm,
                                                              isMultiEncoder=isMulti)
            if name is not None:
                estimateJob.add_alias(f"priors/{name}-startind{startSegInd}")
            triphoneFiles.extend(estimateJob.triphoneFiles)
            diphoneFiles.extend(estimateJob.diphoneFiles)
            contextFiles.extend(estimateJob.contextFiles)
            numSegments.extend(estimateJob.numSegments)



    comJobs = []
    for spliter in range(0, len(triphoneFiles), step):
        start = spliter
        end = spliter + step
        if end > len(triphoneFiles):
            end = triphoneFiles
        comJobs.append(CombineMeansForTriphoneForward(triphoneFiles[start:end],
                                                      diphoneFiles[start:end],
                                                      contextFiles[start:end],
                                                      numSegments[start:end],
                                                      nContexts=nContexts,
                                                      nStates=nStateClasses,
                                                      ))

    combTriphoneFiles = [c.triphoneFilesOut for c in comJobs]
    combDiphoneFiles = [c.diphoneFilesOut for c in comJobs]
    combContextFiles = [c.contextFilesOut for c in comJobs]
    combNumSegs = [c.numSegmentsOut for c in comJobs]
    xmlJob = DumpXmlForTriphoneForward(combTriphoneFiles,
                                       combDiphoneFiles,
                                       combContextFiles,
                                       combNumSegs,
                                       nContexts=nContexts,
                                       nStates=nStateClasses)

    priorFilesTriphone = [xmlJob.triphoneXml, xmlJob.diphoneXml, xmlJob.contextXml]
    xmlName = f"priors/{name}"
    tk.register_output(xmlName, priorFilesTriphone[0])


    return priorFilesTriphone