package ru.msu.cs.sphere;

import ru.msu.cs.sphere.App.*;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AppTest
{
    @Test
    public void testInput() throws Exception
    {
        Reader reader = new Reader();

        String testDataInp1 =
                "2015-07-10 11:30:28 +0300;0;1;390\n" +
                "2015-07-10 11:32:28 +0300;0;1;391\n" +
                "2015-07-10 11:33:28 +0300;0;0;390\n" +
                "2015-07-10 11:34:28 +0300;0;0;391\n" +
                "2015-07-10 11:35:28 +0300;0;1;392\n" +
                "2015-07-10 11:36:28 +0300;0;1;393\n" +
                "2015-07-10 11:37:28 +0300;0;0;392\n" +
                "2015-07-10 11:38:28 +0300;0;0;393";

        String testDataInp =
                "2015-07-10 11:30:28 +0300;взрослый;1;390\n" +
                "2015-07-10 11:32:28 +0300;детский;1;391\n" +
                "2015-07-10 11:33:28 +0300;взрослый;0;390";

        LogEntity log1 = new LogEntity();
        LogEntity log2 = new LogEntity();
        LogEntity log3 = new LogEntity();

        log1.date = App.FORMATTER.parse("2015-07-10 11:30:28 +0300");
        log1.ticketType = "взрослый";
        log1.action = Boolean.TRUE;
        log1.userId = "390";

        log2.date = App.FORMATTER.parse("2015-07-10 11:32:28 +0300");
        log2.ticketType = "детский";
        log2.action = Boolean.TRUE;
        log2.userId = "391";

        log3.date = App.FORMATTER.parse("2015-07-10 11:33:28 +0300");
        log3.ticketType = "взрослый";
        log3.action = Boolean.FALSE;
        log3.userId = "390";

        ArrayList<LogEntity> arrayTest = new ArrayList<LogEntity>();
        arrayTest.add(log1);
        arrayTest.add(log2);
        arrayTest.add(log3);

        List<LogEntity> arrayTarget = reader.getData(new ByteArrayInputStream(testDataInp.getBytes(StandardCharsets.UTF_8)));

        for (int i = 0; i < arrayTarget.size(); i++) {
            LogEntity testLog = arrayTest.get(i);
            LogEntity targetLog = arrayTarget.get(i);
            assertEquals(testLog.date, targetLog.date);
            assertEquals(testLog.ticketType, targetLog.ticketType);
            assertEquals(testLog.action, targetLog.action);
            assertEquals(testLog.userId, targetLog.userId);
        }
    }


//    @Test
//    public void testLogEntityComparison()
//    {
//        LogEntityComparator comparator = new LogEntityComparator();
//        LogEntity log1 = new LogEntity();
//        LogEntity log2 = new LogEntity();
//
//        log1.date = LocalDate.parse("2015-07-10 11:30:28 +0300", formatter);
//        log1.ticketType = "взрослый";
//        log1.action = Boolean.TRUE;
//        log1.userId = "390";
//
//        log2.date = LocalDate.parse("2015-07-10 11:32:28 +0300", formatter);
//        log2.ticketType = "детский";
//        log2.action = Boolean.TRUE;
//        log2.userId = "391";
//
//        assertTrue(LogEntityComparator.compare(log1, log2));
//    }


    @Test
    public void testApp()
    {
        assertTrue(true);
    }

    @Test
    public void testApp1()
    {
        assertTrue(true);
    }

    @Test
    public void testSayHello() throws Exception {
        assertEquals(1, App.sayHello());
    }

    enum T {T1, T2, T3};

    public static final Map<String, Integer> KEY_PROTOCOLS;

    static {
        Map<String, Integer> map = new HashMap<String, Integer>();
        map.put("взрослый", 0);
        map.put("детский", 1);
        map.put("льготный", 2);

        KEY_PROTOCOLS = Collections.unmodifiableMap(map);
    }

    static public String[] TICKET_TYPES = {"взрослый", "детский", "льготный"};

    @Test
    public void test1() throws Exception {

        for (String type :TICKET_TYPES)
        {
            System.out.println(type);
        }

        T t = T.values()[0];
        System.out.println(t);

        System.out.println(KEY_PROTOCOLS.get("детский"));

        System.out.println(Byte.parseByte("-1"));

        System.out.println(Byte.parseByte("10")>0);

        String[] array = "2015-07-10 11:38:28+03:00;0;0;393".split(";", 4);

        for(String str: array)
        {
            System.out.println(str);
        }
    }

    @Test
    public void testDate() throws ParseException {
        String date_s = "2015-07-10 11:30:28 +0300";
        SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z");
        Date date = dt.parse(date_s);

        SimpleDateFormat dt1 = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z");
        System.out.println(dt1.format(date));
    }
}
