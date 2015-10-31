package ru.msu.cs.sphere;

import ru.msu.cs.sphere.App.*;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.Assert.assertEquals;

public class AppTest
{

    @Test
    public void smokeTest() throws Exception {
        Reader reader = new Reader();

        String testData =
            "2015-07-10 11:30:28 +0300;взрослый;1;390\n" +
            "2015-07-10 11:32:28 +0300;взрослый;1;391\n" +
            "2015-07-10 11:33:28 +0300;взрослый;0;390\n" +
            "2015-07-10 11:38:28 +0300;взрослый;0;393\n" +
            "2015-07-10 11:34:28 +0300;взрослый;0;391\n" +
            "2015-07-10 11:35:28 +0300;взрослый;1;392\n" +
            "2015-07-10 11:36:28 +0300;взрослый;1;393\n" +
            "2015-07-10 11:37:28 +0300;взрослый;0;392\n";

        List<LogEntity> logEntities = reader.getData(
                new ByteArrayInputStream(testData.getBytes(StandardCharsets.UTF_8))
        );
        assertEquals(logEntities.size(), 8);

        List<Interval> intervals = App.computePopularTimeIntervals(logEntities);

        assertEquals(intervals.size(), 2);
        assertEquals((int)intervals.get(0).ticketsDist.get(App.ADULT_TICKET_TYPE), 2);
    }

    @Test
    public void testInput() throws Exception {
        String testDataInp =
                "2015-07-10 11:30:28 +0300;взрослый;1;390\n" +
                "2015-07-10 11:32:28 +0300;детский;1;391\n" +
                "2015-07-10 11:33:28 +0300;взрослый;0;390";

        LogEntity log1 = new LogEntity();
        LogEntity log2 = new LogEntity();
        LogEntity log3 = new LogEntity();

        log1.date = App.FORMATTER.parse("2015-07-10 11:30:28 +0300");
        log1.ticketType = App.ADULT_TICKET_TYPE;
        log1.action = Boolean.TRUE;
        log1.userId = "390";

        log2.date = App.FORMATTER.parse("2015-07-10 11:32:28 +0300");
        log2.ticketType = App.CHILDISH_TICKET_TYPE;
        log2.action = Boolean.TRUE;
        log2.userId = "391";

        log3.date = App.FORMATTER.parse("2015-07-10 11:33:28 +0300");
        log3.ticketType = App.ADULT_TICKET_TYPE;
        log3.action = Boolean.FALSE;
        log3.userId = "390";

        ArrayList<LogEntity> arrayTest = new ArrayList<LogEntity>();
        arrayTest.add(log1);
        arrayTest.add(log2);
        arrayTest.add(log3);

        Reader reader = new Reader();
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
}
