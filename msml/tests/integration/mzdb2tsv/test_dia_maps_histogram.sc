
import $ivy.`org.scalatest::scalatest:3.2.9`
import org.scalatest._
import java.io.File
import sys.process._

class DiaMapsHistogramTest extends FunSuite {
  test("dia_maps_histogram.sc processes mzdb file correctly") {
    // Create a test mzdb file
    val testFile = new File("test.mzdb")
    testFile.createNewFile()
    
    try {
      // Run the script
      val result = s"JAVA_OPTS='-Djava.library.path=./' ./amm dia_maps_histogram.sc ${testFile.getPath} 0.1 10".!
      
      // Check if tsv was created
      val tsvFile = new File("test.tsv")
      assert(tsvFile.exists(), "TSV file was not created")
      
      // Check tsv content
      val lines = scala.io.Source.fromFile(tsvFile).getLines().toList
      assert(lines.nonEmpty, "TSV file is empty")
      
      // Check header
      val header = lines.head
      assert(header.contains("mz") && header.contains("rt") && header.contains("intensity"),
             "TSV file does not have required columns")
    } finally {
      // Cleanup
      testFile.delete()
      new File("test.tsv").delete()
    }
  }
}

// Run the tests
val test = new DiaMapsHistogramTest()
test.execute()