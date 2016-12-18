


# Task: https://sphere.mail.ru/blog/topic/1791/


# TODO: Map Reduce ипользуя http://rishavrohitblog.blogspot.ru/2014/09/hbase-mapreduce-on-multiple-input-table.html



scan 'test_kopin', {
    'LIMIT' => 5,
    COLUMNS => ['ftest:ftext']
}

scan 'websites_kopin', {
    'LIMIT' => 5,
    COLUMNS => ['info:robots']
}


scan 'websites_kopin', {'LIMIT' => 100, COLUMNS => ['info:site']}


delete 'webpages_kopin', '000060584f05cd7579be9b0898f683a8', 'docs:disabled'
put 'webpages_kopin', '000060584f05cd7579be9b0898f683a8', 'docs:disabled', 'Y', 1479277626574

scan 'webpages_kopin', {
    LIMIT => 5,
    COLUMNS => ['docs:disabled']
}

scan 'webpages_kopin', {LIMIT => 100, COLUMNS => 'docs:url'}


clone_snapshot 'webpages_bak', 'webpages_kopin'
clone_snapshot 'websites_bak', 'websites_kopin'

scan 'webpages_kopin', {
    COLUMNS => ['docs'],
    FILTER => "(SingleColumnValueFilter('docs','url',=,'regexstring:http:.*/4.html$',true,true))",
    LIMIT => 5
}


# 0. Считываем website
# 1. Если info:robots пустой, то для для всех урлов "info:site*" удаляем webpages_kopin|docs:disabled 

# 2. Делим info:robots по newline и запускаем SingleColumnValueFilter




# Считываем сайт и все его урлы и обрабатываем 


# На маппер: сайт с robots и его урлы


# Заюзать
# http://rishavrohitblog.blogspot.ru/2014/09/hbase-mapreduce-on-multiple-input-table.html



Connection connection = ConnectionFactory.createConnection(conf);

Table table = connection.getTable(TableName.valueOf("webpages_kopin"));
Put put = new Put(Bytes.toBytes("Person1"));
put.addColumn(Bytes.toBytes("meta"), Bytes.toBytes("OS"), Bytes.toBytes("Mac OS X"));
put.addColumn(Bytes.toBytes("meta"), Bytes.toBytes("Browser"), Bytes.toBytes("Chrome"));
table.put(put);

table.close();
connection.close();



RegexStringComparator comp = new RegexStringComparator("http:.*/4.html$");

Scan scan = new Scan()\
    .setFilter(
        new SingleColumnValueFilter(
            Bytes.toBytes("docs"),
            Bytes.toBytes("url"),
            CompareFilter.CompareOp.EQUAL,
            comp
        )
    );


while (scanner.advance()) {
   Cell cell = scanner.current();
   // do something
 }




Configuration config = HBaseConfiguration.create();
HTable table = new HTable(config, "webpages_kopin");

RegexStringComparator comp = new RegexStringComparator("http:.*/4.html$");
Scan scan = new Scan()\
    .setFilter(
        new SingleColumnValueFilter(
            Bytes.toBytes("docs"),
            Bytes.toBytes("url"),
            CompareFilter.CompareOp.EQUAL,
            comp
        )
    );

ResultScanner scanner = table.getScanner(scan);

CellScanner scanner = res.cellScanner();

while (scanner.advance()) {
    Cell cell = scanner.current();
    String qualifier = Bytes.toString(CellUtil.cloneQualifier(cell));
    String value = Bytes.toString(CellUtil.cloneValue(cell));
    System.out.printf("++ Qualifier: %s : Value: %s\n", qualifier, value);
}




export HADOOP_CLASSPATH=${JAVA_HOME}/lib/tools.jar
export CLASSPATH=/usr/lib/hadoop-mapreduce/\*:/usr/lib/hadoop/\*;
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$(hbase classpath)

hadoop com.sun.tools.javac.Main HW4.java
jar cf hw4.jar *.class
hadoop jar hw4.jar HW4 webpages_kopin websites_kopin