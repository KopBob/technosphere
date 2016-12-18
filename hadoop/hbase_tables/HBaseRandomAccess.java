/*
 * 1. setup your hbase client properly (make hbase shell work from console)
 *    setup ZooKeeper addr in hbase-site.xml
 * 2. Launch via:
 *    $ CLASSPATH="`hbase classpath`:out/artifacts/HBaseExamples/HBaseExamples.jar" java HBaseRandomAccess scan 2>/dev/null
 */
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.*;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.util.Bytes;

import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class HBaseRandomAccess {
    void DemoPut() throws IOException {
        Configuration conf = HBaseConfiguration.create();
        Connection connection = ConnectionFactory.createConnection(conf);

        Table table = connection.getTable(TableName.valueOf("btest"));
        Put put = new Put(Bytes.toBytes("Person1"));
        put.addColumn(Bytes.toBytes("meta"), Bytes.toBytes("OS"), Bytes.toBytes("Mac OS X"));
        put.addColumn(Bytes.toBytes("meta"), Bytes.toBytes("Browser"), Bytes.toBytes("Chrome"));
        table.put(put);

        table.close();
        connection.close();
    }

    void DemoGet() throws IOException {
        Configuration conf = HBaseConfiguration.create();
        Connection connection = ConnectionFactory.createConnection(conf);

        Table table = connection.getTable(TableName.valueOf("btest"));

        Get get = new Get(Bytes.toBytes("Person1"));
        Result res = table.get(get);

        /* write actual 'get' here (assumed Result res) */
        for (Cell cell : res.listCells()) {
            String family = Bytes.toString(CellUtil.cloneFamily(cell));
            String qualifier = Bytes.toString(CellUtil.cloneQualifier(cell));
            String value = Bytes.toString(CellUtil.cloneValue(cell));
            System.out.printf("Family: %s Qualifier: %s : Value: %s\n", family, qualifier, value);
        }
        /*CellScanner scanner = res.cellScanner();
        while (scanner.advance()) {
            Cell cell = scanner.current();
            String qualifier = Bytes.toString(CellUtil.cloneQualifier(cell));
            String value = Bytes.toString(CellUtil.cloneValue(cell));
            System.out.printf("++ Qualifier: %s : Value: %s\n", qualifier, value);
        }*/

        table.close();
        connection.close();
    }

    void PrintCell(Cell cell) {
        String qualifier = Bytes.toString(CellUtil.cloneQualifier(cell));
        String value = Bytes.toString(CellUtil.cloneValue(cell));
        System.out.printf("\t%s=%s\n", qualifier, value);
    }

    void PrintResult(Result res) throws UnsupportedEncodingException {
        System.out.printf("------------- ROW: %s\n", new String(res.getRow(), "UTF8"));
        for (Cell cell : res.listCells())
            PrintCell(cell);
    }

    void DemoScan() throws IOException {
        Configuration conf = HBaseConfiguration.create();
        Connection connection = ConnectionFactory.createConnection(conf);

        /* write your code here using PrintResult */
        connection.close();
    }

    public static void main(String[] args) throws IOException {
        HBaseRandomAccess demo = new HBaseRandomAccess();

        if (args.length != 1)
            usage();

        if (args[0].equals("put"))
            demo.DemoPut();
        else if (args[0].equals("get"))
            demo.DemoGet();
        else if (args[0].equals("scan"))
            demo.DemoScan();
        else
            usage();
    }

    static void usage() {
        System.err.println("wrong usage!");
        System.exit(64);
    }
}
