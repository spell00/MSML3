
interp.load.cp(ammonite.ops.Path("./mzdb-access-0.8.0.jar", os.pwd))

@

// mzdb-access dependencies
import $ivy.`ch.qos.logback:logback-classic:1.2.3`
import $ivy.`org.apache.commons:commons-lang3:3.6`
import $ivy.`com.almworks.sqlite4java:sqlite4java:1.0.392`
import $ivy.`com.almworks.sqlite4java:libsqlite4java-linux-amd64:1.0.392`
import $ivy.`com.beust:jcommander:1.72`
import $ivy.`xerces:xercesImpl:2.11.0`

import $ivy.`io.reactivex:rxjava:1.3.0`
import $ivy.`io.reactivex:rxscala_2.11:0.26.5`

// Import classes
import java.io.File
import java.io.PrintWriter
import scala.collection.mutable.LongMap
import fr.profi.mzdb._

@main
def main(path: os.Path, mzBinSize: String = "0.1", rtBinSize: String = "340"): Unit = {

  val inputFile = path.toIO
  val inputNameHead = inputFile.getName.split('.').head

  val ms1OutputFile = new File ( s"${inputNameHead}.tsv")

  create_ms1_histogram(inputFile, ms1OutputFile, mzBinSize.toFloat, rtBinSize.toFloat)
/*
  val msnOutputFile = new File ( s"${inputNameHead}_dia_densities.tsv")

  create_dia_histogram(inputFile, msnOutputFile, mzBinSize.toFloat, rtBinSize.toFloat)*/
}

def create_ms1_histogram(mzDbFile: File, outputFile: File, mzBinSize: Float, rtBinSize: Float) {

  println(s"Processing MS1 data...")

  val printWriter = new PrintWriter(outputFile)
  printWriter.println(List("rt_bin","mz_bin","bin_intensity").mkString("\t"))

  val mzDb = new MzDbReader(mzDbFile, true)

  val rsIter = mzDb.getLcMsRunSliceIterator

  val ticByMzAndRtIdx = new LongMap[LongMap[Float]]()
  while (rsIter.hasNext()) {

    val runSlice = rsIter.next()
    val rsData = runSlice.getData()
    val spectrumSlices = rsData.getSpectrumSliceList()

    for(
      spectrumSlice <- spectrumSlices;
      rt = spectrumSlice.getHeader.getElutionTime;
      ticByMzIdx = ticByMzAndRtIdx.getOrElseUpdate( (rt / rtBinSize).toInt, new LongMap[Float]() );
      peak <- spectrumSlice.toPeaks()
    ) {
      val peakMzIdx = (peak.getMz() / mzBinSize).toInt

      // Update TIC value
      val newTic = ticByMzIdx.get(peakMzIdx).map( _ + peak.getIntensity() ).getOrElse(0f)
      ticByMzIdx( peakMzIdx ) = newTic
    }
  }

  val rtIndices = ticByMzAndRtIdx.keys.toArray.sorted

  for (
    rtIdx <- rtIndices;
    ticByMzIdx = ticByMzAndRtIdx(rtIdx);
    mzIdx <- ticByMzIdx.keys.toArray.sorted
  ) {
    printWriter.println( rtIdx * rtBinSize + "\t" + mzIdx * mzBinSize + "\t" + ticByMzIdx(mzIdx) )
  }

  printWriter.close()

  mzDb.close()
}

def create_dia_histogram(mzDbFile: File, outputFile: File, mzBinSize: Float, rtBinSize: Float) {

  println(s"Processing MSn data...")

  val printWriter = new PrintWriter(outputFile)
  printWriter.println(List("min_parent_mz","max_parent_mz","rt_bin","mz_bin","bin_intensity").mkString("\t"))

  val mzDb = new MzDbReader(mzDbFile, true)
  val diaWindows = mzDb.getDIAIsolationWindows()
  println(s"Found ${diaWindows.length} DIA windows")

  for (diaWindow <- diaWindows) {
    val rsIter = mzDb.getLcMsnRunSliceIterator(diaWindow.getMinMz(),diaWindow.getMaxMz())

    val ticByMzAndRtIdx = new LongMap[LongMap[Float]]()
    while (rsIter.hasNext()) {

      val runSlice = rsIter.next()
      val rsData = runSlice.getData()
      val spectrumSlices = rsData.getSpectrumSliceList()

      for(
        spectrumSlice <- spectrumSlices;
        rt = spectrumSlice.getHeader.getElutionTime;
        ticByMzIdx = ticByMzAndRtIdx.getOrElseUpdate( (rt / rtBinSize).toInt, new LongMap[Float]() );
        peak <- spectrumSlice.toPeaks()
      ) {
        val peakMzIdx = (peak.getMz() / mzBinSize).toInt

        // Update TIC value
        val newTic = ticByMzIdx.get(peakMzIdx).map( _ + peak.getIntensity() ).getOrElse(0f)
        ticByMzIdx( peakMzIdx ) = newTic
      }
    }

    for (
      rtIdx <- ticByMzAndRtIdx.keys.toArray.sorted;
      ticByMzIdx = ticByMzAndRtIdx(rtIdx);
      mzIdx <- ticByMzIdx.keys.toArray.sorted
    ) {
      printWriter.println( diaWindow.getMinMz+ "\t" +diaWindow.getMaxMz + "\t" + rtIdx * rtBinSize + "\t" + mzIdx * mzBinSize + "\t" + ticByMzIdx(mzIdx) )
    }
    println("Exported DIA window: " + diaWindow.getMinMz + "\t" +diaWindow.getMaxMz)
  }

  printWriter.close()

  mzDb.close()
}
